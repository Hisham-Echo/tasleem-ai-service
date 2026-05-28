import re
import pickle
import logging
import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import asynccontextmanager
from rapidfuzz import process, fuzz


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ml_models = {}


# ============================================================
# ELECTRONICS QUERY EXPANSION DICTIONARY
# ============================================================
ELECTRONICS_SYNONYMS = {
    # Typos + synonyms
    "moble":    ["mobile", "phone", "smartphone"],
    "moblie":   ["mobile", "phone", "smartphone"],
    "phon":     ["phone", "smartphone", "mobile"],
    "labtop":   ["laptop", "notebook", "computer"],
    "leptop":   ["laptop", "notebook"],
    "camra":    ["camera", "cam"],
    "headphon": ["headphone", "earphone", "headset"],
    "earbud":   ["earbuds", "earphone", "tws", "wireless earphone"],
    "tv":       ["television", "smart tv", "led tv", "oled"],
    "ac":       ["air conditioner", "air conditioning", "cooling"],
    "pc":       ["computer", "desktop", "personal computer"],
    "tab":      ["tablet", "ipad"],
    "watch":    ["smartwatch", "smart watch", "wearable"],
    "charge":   ["charger", "charging", "adapter", "power bank"],
    "bluetooth":["wireless", "bt", "wifi"],
    "gaming":   ["game", "gamer", "console", "playstation", "xbox"],
    "apple":    ["iphone", "ipad", "macbook", "airpods"],
    "samsung":  ["galaxy", "samsung phone", "samsung tv"],
    "sony":     ["playstation", "sony camera", "sony headphone"],
}


# ─────────────────────────────────────────────
# LIFESPAN — load all models once at startup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading all models...")

    def safe_load(name: str, loader):
        try:
            ml_models[name] = loader()
            logger.info(f"✓ Loaded: {name}")
        except Exception as e:
            logger.error(f"✗ Failed to load [{name}]: {e}")
            ml_models[name] = None

    safe_load("index",          lambda: faiss.read_index("faiss_last.index"))
    safe_load("id_map",         lambda: pickle.load(open("id_map_last.pkl", "rb")))
    safe_load("reverse_id_map", lambda: pickle.load(open("reverse_id_map_last.pkl", "rb")))
    safe_load("popular",        lambda: pickle.load(open("popular_last.pkl", "rb")))
    safe_load("trending",       lambda: pickle.load(open("trending_last.pkl", "rb")))
    safe_load("user_recs",      lambda: pickle.load(open("hybrid_recs_last_v2.pkl", "rb")))
    safe_load("products",       lambda: pickle.load(open("products_last.pkl", "rb")))

    try:
        tfidf_data = pickle.load(open("tfidf_search_last.pkl", "rb"))
        ml_models["tfidf_vectorizer"] = tfidf_data[0]
        ml_models["tfidf_matrix"]     = tfidf_data[1]
        ml_models["tfidf_ids"]        = tfidf_data[2]
        logger.info("✓ Loaded: tfidf_search")
    except Exception as e:
        logger.error(f"✗ Failed to load tfidf_search: {e}")
        ml_models["tfidf_vectorizer"] = None
        ml_models["tfidf_matrix"]     = None
        ml_models["tfidf_ids"]        = None

    safe_load("sentiment_summary", lambda: pickle.load(open("sentiment_summary_last_v2.pkl", "rb")))
    safe_load("bundles",           lambda: pickle.load(open("bundles.pkl", "rb")))

    if ml_models.get("tfidf_vectorizer") is None:
        logger.warning("TF-IDF not loaded from file — building from products...")
        build_tfidf_from_products()

    build_search_vocabulary()

    logger.info("All models loaded. Server is ready.")
    yield
    ml_models.clear()
    logger.info("Shutdown complete — memory cleared.")


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="Tasleem AI Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: lock down in production
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────
# TF-IDF BUILDER — fallback if pkl missing
# ─────────────────────────────────────────────

def build_tfidf_from_products():
    products = ml_models.get("products")
    if products is None:
        logger.error("Cannot build TF-IDF — products not loaded.")
        return

    if hasattr(products, "iterrows"):
        items = [(row["id"], row) for _, row in products.iterrows()]
    else:
        items = [(pid, p) for pid, p in products.items()]

    ids, texts = [], []
    for pid, product in items:
        ids.append(pid)
        text = " ".join(filter(None, [
            str(product.get("name", "")),
            str(product.get("description", "")),
            str(product.get("category", "")),
            str(product.get("brand", "")),
        ]))
        texts.append(text.lower())

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix     = vectorizer.fit_transform(texts)

    ml_models["tfidf_vectorizer"] = vectorizer
    ml_models["tfidf_matrix"]     = matrix
    ml_models["tfidf_ids"]        = ids
    logger.info(f"✓ TF-IDF built from products: {len(ids)} docs, {matrix.shape[1]} features")


# ─────────────────────────────────────────────
# SEARCH VOCABULARY — for typo correction
# ─────────────────────────────────────────────

def build_search_vocabulary():
    products = ml_models.get("products")
    if products is None:
        ml_models["search_vocab"] = []
        return

    vocab = set()
    rows = [row for _, row in products.iterrows()] if hasattr(products, "iterrows") else list(products.values())

    for product in rows:
        for field in ["name", "brand", "category"]:
            value = product.get(field, "")
            if value:
                for word in str(value).lower().split():
                    word = re.sub(r"[^a-z0-9\-]", "", word)
                    if len(word) > 2 and not re.search(r"\d{2}:\d{2}", word):
                        vocab.add(word)

    ml_models["search_vocab"] = list(vocab)
    logger.info(f"✓ Search vocabulary built: {len(vocab)} words")


# ─────────────────────────────────────────────
# LAYER 0 — TYPO CORRECTION
# ─────────────────────────────────────────────

def correct_query(query: str, threshold: int = 70) -> str:
    """Fix typos word-by-word against the product vocabulary."""
    vocab = ml_models.get("search_vocab", [])
    if not vocab:
        return query

    corrected_words = []
    for word in query.lower().split():
        if len(word) <= 2 or word.isdigit():
            corrected_words.append(word)
            continue
        if word in vocab:
            corrected_words.append(word)
            continue

        match = process.extractOne(word, vocab, scorer=fuzz.ratio, score_cutoff=threshold)
        if match:
            logger.info(f"Typo corrected: '{word}' → '{match[0]}' (score={match[1]})")
            corrected_words.append(match[0])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# ─────────────────────────────────────────────
# LAYER 1 — QUERY EXPANSION
# ─────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """Expand query using synonyms/abbreviations dictionary."""
    q = query.lower().strip()
    terms = [q]

    if q in ELECTRONICS_SYNONYMS:
        terms.extend(ELECTRONICS_SYNONYMS[q])

    for key, vals in ELECTRONICS_SYNONYMS.items():
        if key in q or q in key:
            terms.extend(vals)

    return list(dict.fromkeys(terms))  # dedupe, preserve order


# ─────────────────────────────────────────────
# LAYER 2 — TF-IDF SEARCH (single query)
# ─────────────────────────────────────────────

def search_ids(q: str, k: int = 10) -> list:
    """Basic TF-IDF cosine similarity search."""
    try:
        vectorizer = ml_models.get("tfidf_vectorizer")
        matrix     = ml_models.get("tfidf_matrix")
        t_ids      = ml_models.get("tfidf_ids")

        if vectorizer is None:
            return []

        q_vec   = vectorizer.transform([q])
        scores  = cosine_similarity(q_vec, matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        return [t_ids[i] for i in top_idx if scores[i] > 0]
    except Exception as e:
        logger.error(f"Error in search_ids: {e}")
        return []


# ─────────────────────────────────────────────
# LAYER 3 — FAISS SIMILARITY EXPANSION
# ─────────────────────────────────────────────

def similar_ids(product_id: int, k: int = 10) -> list:
    """Find similar products using FAISS vector index."""
    try:
        rev_map = ml_models.get("reverse_id_map", {})
        idx     = ml_models.get("index")
        id_m    = ml_models.get("id_map")

        if not rev_map or product_id not in rev_map or idx is None:
            return []

        vec = idx.reconstruct(rev_map[product_id]).reshape(1, -1)
        _, I = idx.search(vec, k + 1)
        return [id_m[i] for i in I[0] if id_m[i] != product_id][:k]
    except Exception as e:
        logger.error(f"Error in similar_ids: {e}")
        return []


# ─────────────────────────────────────────────
# CORE — FULL 4-LAYER SEARCH ENGINE
# ─────────────────────────────────────────────

def full_search(query: str, k: int = 10) -> list:
    """
    Full pipeline:
      Layer 0 → Typo correction       (correct_query)
      Layer 1 → Query expansion       (expand_query)
      Layer 2 → TF-IDF on all variants (search_ids)
      Layer 3 → Fuzzy match on titles  (rapidfuzz WRatio)
      Layer 4 → FAISS vector search    (similar_ids via index)
      Merge all scores → return top-k
    """
    scores: dict = {}  # pid -> best_score

    # ── Layer 0: Typo correction ────────────────────────────────
    corrected = correct_query(query)
    if corrected != query:
        logger.info(f"Query corrected: '{query}' → '{corrected}'")

    # ── Layer 1: Query expansion ────────────────────────────────
    all_queries = expand_query(corrected)
    # Fallback: also expand original in case correction changed meaning
    if corrected != query:
        for term in expand_query(query):
            if term not in all_queries:
                all_queries.append(term)

    logger.info(f"Expanded queries: {all_queries}")

    # ── Layer 2: TF-IDF on all query variants ───────────────────
    vectorizer = ml_models.get("tfidf_vectorizer")
    matrix     = ml_models.get("tfidf_matrix")
    tfidf_ids  = ml_models.get("tfidf_ids")

    if vectorizer is not None:
        for q in all_queries:
            try:
                q_vec = vectorizer.transform([q])
                sims  = (matrix @ q_vec.T).toarray().flatten()
                top_idx = sims.argsort()[::-1][:k * 2]
                for idx in top_idx:
                    if sims[idx] > 0.01:
                        pid = tfidf_ids[idx]
                        scores[pid] = max(scores.get(pid, 0), float(sims[idx]))
            except Exception as e:
                logger.warning(f"TF-IDF layer error for '{q}': {e}")

    # ── Layer 3: Fuzzy match on product titles ───────────────────
    products = ml_models.get("products")
    if products is not None:
        try:
            if hasattr(products, "iterrows"):
                pid_titles = [
                    (row["id"], str(row.get("name", "")) + " " + str(row.get("category", "")))
                    for _, row in products.iterrows()
                ]
            else:
                pid_titles = [
                    (pid, str(p.get("name", "")) + " " + str(p.get("category", "")))
                    for pid, p in products.items()
                ]

            title_strings = [t for _, t in pid_titles]

            fuzzy_hits = process.extract(
                corrected,          # use corrected query for fuzzy
                title_strings,
                scorer=fuzz.WRatio,
                limit=k * 3,
                score_cutoff=45
            )
            for _, score, idx in fuzzy_hits:
                pid = pid_titles[idx][0]
                normalized = (score / 100.0) * 0.8   # scale to 0–0.8
                scores[pid] = max(scores.get(pid, 0), normalized)

        except Exception as e:
            logger.warning(f"Fuzzy layer error: {e}")

    # ── Layer 4: FAISS vector search ─────────────────────────────
    faiss_index = ml_models.get("index")
    id_map      = ml_models.get("id_map")

    if faiss_index is not None and vectorizer is not None:
        for q in all_queries[:3]:   # top 3 variants only (speed)
            try:
                q_vec = vectorizer.transform([q]).toarray().astype("float32")
                norm  = np.linalg.norm(q_vec)
                if norm > 0:
                    q_vec /= norm
                D, I = faiss_index.search(q_vec, k * 2)
                for dist, idx in zip(D[0], I[0]):
                    if 0 <= idx < len(id_map):
                        pid = id_map[idx]
                        sim = float(dist) if dist > 0 else 1.0 / (1.0 + abs(dist))
                        scores[pid] = max(scores.get(pid, 0), sim * 0.9)
            except Exception as e:
                logger.warning(f"FAISS layer error for '{q}': {e}")

    # ── Merge + sort ─────────────────────────────────────────────
    if not scores:
        # Last resort: word-by-word TF-IDF
        for word in corrected.split():
            candidates = search_ids(word, k=8)
            if candidates:
                for i, pid in enumerate(candidates):
                    scores[pid] = max(scores.get(pid, 0), 0.1 / (i + 1))
                break

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [pid for pid, _ in ranked[:k]]

    logger.info(f"Search '{query}' → {len(result)} results")
    return result


# Keep semantic_search as alias for backward compatibility
semantic_search = full_search


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    products = ml_models.get("products")
    return {
        "status": "online",
        "products_loaded": len(products) if products is not None else 0,
        "sentiment_summary": ml_models.get("sentiment_summary") is not None,
        "tfidf_ready": ml_models.get("tfidf_vectorizer") is not None,
        "vocab_size": len(ml_models.get("search_vocab", [])),
    }


@app.get("/recommend/user/{user_id}")
async def user_recommendations(user_id: int, last_product_id: int = None, k: int = 20):
    recs = ml_models.get("user_recs", {}) or {}

    if user_id in recs and recs[user_id]:
        return {"section": "For You", "ids": recs[user_id][:k]}

    if last_product_id:
        sim = similar_ids(last_product_id, k)
        if sim:
            return {"section": "Based on your last view", "ids": sim}

    popular = ml_models.get("popular") or []
    return {"section": "Popular Products", "ids": popular[:k]}


@app.get("/trending")
def get_trending(k: int = 20):
    return {"section": "Trending Now", "ids": (ml_models.get("trending") or [])[:k]}


@app.get("/explore")
def get_explore(k: int = 20):
    products = ml_models.get("products")
    if products is None:
        return {"ids": []}
    # Support both dict and DataFrame
    if hasattr(products, "sample"):
        return {"section": "Explore More", "ids": products.sample(min(k, len(products)))["id"].tolist()}
    else:
        import random
        all_ids = list(products.keys())
        return {"section": "Explore More", "ids": random.sample(all_ids, min(k, len(all_ids)))}


@app.get("/similar/{product_id}")
def get_similar(product_id: int, k: int = 20):
    return {"ids": similar_ids(product_id, k)}


@app.get("/search")
async def search(q: str = Query(..., min_length=1, max_length=200), k: int = 20):
    return {"ids": full_search(q, k)}


@app.get("/debug/search")
async def debug_search(q: str = Query(...)):
    vocab     = ml_models.get("search_vocab", [])
    word      = q.lower()
    in_vocab  = word in vocab
    match     = process.extractOne(word, vocab, scorer=fuzz.ratio, score_cutoff=50)
    corrected = correct_query(q)
    expanded  = expand_query(corrected)
    results   = full_search(q, k=5)

    return {
        "original_query":  q,
        "corrected_query": corrected,
        "expanded_queries": expanded,
        "word_in_vocab":   in_vocab,
        "fuzzy_match":     match,
        "vocab_size":      len(vocab),
        "vocab_sample":    vocab[:20],
        "search_results":  results,
    }


@app.get("/debug")
async def debug():
    return {
        "models_keys":     list(ml_models.keys()),
        "has_vectorizer":  ml_models.get("tfidf_vectorizer") is not None,
        "has_matrix":      ml_models.get("tfidf_matrix") is not None,
        "has_tfidf_ids":   ml_models.get("tfidf_ids") is not None,
        "tfidf_ids_count": len(ml_models.get("tfidf_ids") or []),
        "has_index":       ml_models.get("index") is not None,
        "has_id_map":      ml_models.get("id_map") is not None,
        "has_reverse_map": ml_models.get("reverse_id_map") is not None,
        "vocab_size":      len(ml_models.get("search_vocab", [])),
    }


# ─────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────

@app.get("/reviews/summary/{product_id}")
def product_sentiment_summary(product_id: int):
    summary = ml_models.get("sentiment_summary") or {}
    result  = summary.get(product_id) or summary.get(str(product_id))

    if not result:
        return {
            "product_id": product_id,
            "total_reviews": 0,
            "positive": 0, "positive_pct": 0,
            "neutral":  0, "neutral_pct":  0,
            "negative": 0, "negative_pct": 0,
            "overall":  "unknown",
            "avg_confidence": 0,
            "sample_reviews": {"positive": [], "negative": [], "neutral": []}
        }
    return result


# ─────────────────────────────────────────────
# BUNDLES
# ─────────────────────────────────────────────

@app.get("/bundle/{product_id}")
def get_bundle(product_id: int, k: int = 10):
    bundles = ml_models.get("bundles") or {}

    if product_id in bundles and bundles[product_id]:
        return {
            "product_id": product_id,
            "section":    "Complete the Setup",
            "ids":        bundles[product_id][:k],
            "source":     "association_rules"
        }
    return {
        "product_id": product_id,
        "section":    "You Might Also Like",
        "ids":        similar_ids(product_id, k),
        "source":     "similarity_fallback"
    }




# ─────────────────────────────────────────────
#  RAILWAY COMPATIBLE STARTUP 
# ─────────────────────────────────────────────

import os

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting Tasleem AI Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


import os
import io
import re
import json
import time
import pickle
import logging
import threading
import faiss
import requests
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from contextlib import asynccontextmanager
from rapidfuzz import process, fuzz

# TFLite runtime for the electronic-image detector (lightweight LiteRT; falls
# back to tflite_runtime or full TensorFlow if LiteRT isn't present).
try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter as TFLiteInterpreter
        except Exception:
            TFLiteInterpreter = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Live backend (Laravel API). Lets the AI serve products created AFTER the
# offline models were trained (new listings become searchable + similar-able).
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://tasleembackendapi-production.up.railway.app/api/v1",
).rstrip("/")
# Optional token protecting POST /refresh (open when unset).
REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN", "")
# Re-sync the live catalog every N hours in the background (0 disables).
SYNC_INTERVAL_HOURS = float(os.environ.get("SYNC_INTERVAL_HOURS", "6"))
# How long (seconds) a live-classified sentiment summary stays cached per product.
SENTIMENT_TTL = float(os.environ.get("SENTIMENT_TTL_SECONDS", "3600"))

ml_models = {}

# product_id -> (timestamp, summary_dict) for live sentiment results.
_sentiment_cache = {}
_sentiment_lock = threading.Lock()

# TFLite Interpreter is NOT thread-safe — serialise inference calls.
_electronic_lock = threading.Lock()
ELECTRONIC_MODEL_PATH = os.environ.get("ELECTRONIC_MODEL_PATH", "electronic_detector.tflite")
ELECTRONIC_LABELS_PATH = os.environ.get("ELECTRONIC_LABELS_PATH", "labels.json")


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
    "airpods":  ["earbuds", "wireless earphone", "apple"],
    "ps5":      ["playstation 5", "console"],
    "ps4":      ["playstation 4", "console"],
    "xbox":     ["console", "series x", "series s"],
    "macbook":  ["laptop", "apple"],
    "ssd":      ["storage", "solid state", "hard drive"],
    "hdd":      ["hard drive", "storage"],
    "powerbank":["power bank", "charger", "battery"],

    # Arabic → English (what users actually type)
    "موبايل":     ["mobile", "phone", "smartphone"],
    "موبيل":      ["mobile", "phone", "smartphone"],
    "تليفون":     ["phone", "smartphone"],
    "هاتف":       ["phone", "smartphone"],
    "لابتوب":     ["laptop", "notebook"],
    "لاب توب":    ["laptop", "notebook"],
    "كمبيوتر":    ["computer", "desktop", "pc"],
    "سماعة":      ["headphones", "earbuds", "speaker"],
    "سماعات":     ["headphones", "earbuds", "speakers"],
    "شاشة":       ["monitor", "screen", "tv"],
    "تلفزيون":    ["tv", "television"],
    "كاميرا":     ["camera"],
    "ساعة":       ["watch", "smartwatch"],
    "شاحن":       ["charger", "power bank"],
    "تابلت":      ["tablet", "ipad"],
    "ايفون":      ["iphone", "apple"],
    "ايباد":      ["ipad", "tablet"],
    "بلايستيشن":  ["playstation", "console"],
    "راوتر":      ["router", "wifi"],
    "برنتر":      ["printer"],
    "طابعة":      ["printer"],
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
    # Trained sentiment classifier (sklearn Pipeline: raw text -> positive/neutral/negative).
    # Enables LIVE classification of new comments; falls back to the baked summary if absent.
    safe_load("sentiment_model",   lambda: pickle.load(open("sentiment_model_last_v2.pkl", "rb")))
    safe_load("bundles",           lambda: pickle.load(open("bundles.pkl", "rb")))

    # Electronic-image detector (TFLite) — used by POST /detect/electronic to
    # block non-electronic product photos at listing time.
    try:
        if TFLiteInterpreter is None:
            raise RuntimeError("no TFLite runtime installed")
        labels = json.load(open(ELECTRONIC_LABELS_PATH, "r", encoding="utf-8"))
        interp = TFLiteInterpreter(model_path=ELECTRONIC_MODEL_PATH)
        interp.allocate_tensors()
        ml_models["electronic_interp"] = interp
        ml_models["electronic_labels"] = labels
        logger.info(f"✓ Loaded: electronic_detector ({len(labels.get('classes', []))} classes)")
    except Exception as e:
        logger.error(f"✗ Failed to load electronic_detector: {e}")
        ml_models["electronic_interp"] = None
        ml_models["electronic_labels"] = None

    if ml_models.get("tfidf_vectorizer") is None:
        logger.warning("TF-IDF not loaded from file — building from products...")
        build_tfidf_from_products()

    build_search_vocabulary()

    # Pull the LIVE catalog in the BACKGROUND so startup stays fast (the
    # server binds immediately and serves the snapshot until the sync lands).
    # New listings work without a retrain; failure is non-fatal.
    def _initial_sync():
        try:
            synced = sync_live_catalog()
            logger.info(f"✓ Live sync after startup: {synced} products from backend")
        except Exception as e:
            logger.error(f"✗ Live sync failed (continuing with snapshot): {e}")
    threading.Thread(target=_initial_sync, daemon=True).start()

    # Periodic background re-sync.
    if SYNC_INTERVAL_HOURS > 0:
        def _periodic_sync():
            while True:
                time.sleep(SYNC_INTERVAL_HOURS * 3600)
                try:
                    sync_live_catalog()
                except Exception as e:
                    logger.error(f"Periodic live sync failed: {e}")
        threading.Thread(target=_periodic_sync, daemon=True).start()

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
        name = str(product.get("name", ""))
        text = " ".join(filter(None, [
            name, name, name,  # weight the product NAME 3× vs description
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
# LIVE CATALOG SYNC — pull current products from the Laravel backend so
# newly listed items are searchable / similar-able without retraining.
# ─────────────────────────────────────────────

def _get_product(products, product_id):
    """Uniform product lookup for both dict and DataFrame catalogs."""
    if products is None:
        return None
    if hasattr(products, "iterrows"):
        rows = products[products["id"] == product_id]
        return rows.iloc[0].to_dict() if len(rows) else None
    return products.get(product_id)


def _is_sold_out(p) -> bool:
    """Matches the app's availability rule: unavailable status OR no stock.
    (Catches sold-out items even when the backend hasn't flipped status.)"""
    if p is None:
        return False
    if str(p.get("status", "1")) == "0":
        return True
    try:
        return "quantity" in p and int(p.get("quantity") or 0) <= 0
    except (TypeError, ValueError):
        return False


def fetch_live_products(max_pages: int = 100) -> dict:
    """Fetch the full live catalog (paginated). Returns {id: {...}} or {}."""
    out, page = {}, 1
    while page <= max_pages:
        try:
            r = requests.get(
                f"{BACKEND_URL}/products",
                params={"per_page": 100, "page": page},
                timeout=30,
            )
            body = r.json()
        except Exception as e:
            logger.error(f"Live fetch failed on page {page}: {e}")
            break
        for p in body.get("data") or []:
            pid = p.get("id")
            if pid is None:
                continue
            out[pid] = {
                "id": pid,
                "name": p.get("name") or "",
                "description": p.get("description") or "",
                "category": ((p.get("category") or {}).get("name")) or "",
                "brand": "",
                "status": str(p.get("status", "1")),
                "quantity": p.get("quantity") or 0,
                "rate": float(p.get("rate") or 0),
                "view_count": p.get("view_count") or 0,
            }
        pag = body.get("pagination") or {}
        if page >= (pag.get("last_page") or page):
            break
        page += 1
    logger.info(f"Live catalog fetched: {len(out)} products from {BACKEND_URL}")
    return out


def sync_live_catalog() -> int:
    """Merge live products into ml_models['products'] (normalising a snapshot
    DataFrame to a dict) and rebuild the TF-IDF index + search vocabulary."""
    live = fetch_live_products()
    if not live:
        return 0
    merged = {}
    snapshot = ml_models.get("products")
    if snapshot is not None:
        if hasattr(snapshot, "iterrows"):
            for _, row in snapshot.iterrows():
                merged[row["id"]] = dict(row)
        else:
            merged.update(snapshot)
    merged.update(live)  # live data wins on conflicts
    ml_models["products"] = merged
    ml_models["live_sync_count"] = len(live)
    build_tfidf_from_products()
    build_search_vocabulary()
    logger.info(f"✓ Live sync complete: catalog now {len(merged)} products")
    return len(live)


def fetch_live_reviews(product_id: int, max_pages: int = 50) -> list:
    """Fetch all comment strings for a product from the backend (paginated)."""
    out, page = [], 1
    while page <= max_pages:
        try:
            r = requests.get(
                f"{BACKEND_URL}/reviews",
                params={"product_id": product_id, "per_page": 100, "page": page},
                timeout=20,
            )
            body = r.json()
        except Exception as e:
            logger.error(f"Live reviews fetch failed for {product_id} p{page}: {e}")
            break
        for rv in body.get("data") or []:
            comment = (rv.get("comment") or "").strip()
            if comment:
                out.append(comment)
        pag = body.get("pagination") or {}
        if page >= (pag.get("last_page") or page):
            break
        page += 1
    return out


def classify_reviews_live(product_id: int):
    """Run the trained sentiment model over the product's live comments and
    build a summary in the EXACT shape of the baked summaries (so the app needs
    no changes). Returns None if the model is missing or there are no comments."""
    model = ml_models.get("sentiment_model")
    if model is None:
        return None
    comments = fetch_live_reviews(product_id)
    if not comments:
        return None
    try:
        labels = [str(x) for x in model.predict(comments)]
        confs = [float(max(row)) for row in model.predict_proba(comments)]
    except Exception as e:
        logger.error(f"Sentiment classify failed for {product_id}: {e}")
        return None

    total = len(comments)
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    samples = {"positive": [], "negative": [], "neutral": []}
    for comment, label, _conf in zip(comments, labels, confs):
        if label not in counts:
            continue
        counts[label] += 1
        if len(samples[label]) < 3:
            samples[label].append(comment)

    pct = lambda n: round(n * 100.0 / total, 1) if total else 0
    overall = max(counts, key=counts.get) if total else "unknown"
    return {
        "product_id":     product_id,
        "total_reviews":  total,
        "positive":       counts["positive"],
        "neutral":        counts["neutral"],
        "negative":       counts["negative"],
        "positive_pct":   pct(counts["positive"]),
        "negative_pct":   pct(counts["negative"]),
        "neutral_pct":    pct(counts["neutral"]),
        "overall":        overall,
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0,
        "sample_reviews": samples,
        "source":         "live",
    }


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

def tfidf_similar(product_id: int, k: int = 10) -> list:
    """Content-based similar products via TF-IDF cosine similarity.
    Covers products that are NOT in the FAISS index (e.g. newly listed):
    works for anything present in the (live-synced) catalog."""
    try:
        products   = ml_models.get("products")
        vectorizer = ml_models.get("tfidf_vectorizer")
        matrix     = ml_models.get("tfidf_matrix")
        t_ids      = ml_models.get("tfidf_ids")
        p = _get_product(products, product_id)
        if p is None or vectorizer is None or matrix is None:
            return []
        text = " ".join(filter(None, [
            str(p.get("name", "")),
            str(p.get("description", "")),
            str(p.get("category", "")),
        ])).lower()
        q_vec  = vectorizer.transform([text])
        scores = cosine_similarity(q_vec, matrix).flatten()
        out = []
        for i in scores.argsort()[::-1]:
            if scores[i] <= 0 or len(out) >= k:
                break
            pid = t_ids[i]
            if pid == product_id:
                continue
            if _is_sold_out(_get_product(products, pid)):
                continue
            out.append(pid)
        return out
    except Exception as e:
        logger.error(f"Error in tfidf_similar: {e}")
        return []


def similar_ids(product_id: int, k: int = 10) -> list:
    """Find similar products using FAISS vector index."""
    try:
        rev_map = ml_models.get("reverse_id_map", {})
        idx     = ml_models.get("index")
        id_m    = ml_models.get("id_map")

        if not rev_map or product_id not in rev_map or idx is None:
            # Not in the trained index (e.g. newly listed) → content similarity.
            return tfidf_similar(product_id, k)

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

    # ── Re-rank: drop sold-out items, lightly boost well-rated/viewed ones ──
    products = ml_models.get("products")

    def _final_score(pid, base):
        p = _get_product(products, pid)
        if p is None:
            return base
        if _is_sold_out(p):
            return -1.0  # sold out → exclude from results
        boost = 1.0
        boost += min(float(p.get("rate") or 0) / 25.0, 0.2)        # ≤ +20%
        boost += min(float(p.get("view_count") or 0) / 5000.0, 0.1)  # ≤ +10%
        return base * boost

    rescored = ((pid, _final_score(pid, s)) for pid, s in scores.items())
    ranked = sorted((x for x in rescored if x[1] >= 0),
                    key=lambda x: x[1], reverse=True)
    result = [pid for pid, _ in ranked[:k]]

    logger.info(f"Search '{query}' → {len(result)} results")
    return result


# Keep semantic_search as alias for backward compatibility
semantic_search = full_search


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# ELECTRONIC IMAGE DETECTOR  (listing-photo gate)
# ─────────────────────────────────────────────

def classify_electronic_image(image_bytes: bytes) -> dict:
    """Run the TFLite classifier on an uploaded photo and decide whether it's an
    electronic product. Fails OPEN (accept=True) when the model is unavailable so
    a model outage never blocks legitimate sellers."""
    interp = ml_models.get("electronic_interp")
    labels = ml_models.get("electronic_labels")
    if interp is None or labels is None:
        return {"accept": True, "is_electronic": True, "model_available": False,
                "label": None, "category": None, "confidence": 0.0,
                "message": "Image check is unavailable right now."}

    from PIL import Image
    size = int(labels.get("img_size", 224))
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)[None, ...]  # [1,H,W,3] in [0,255]

    with _electronic_lock:
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        x = arr.astype(inp["dtype"]) if inp["dtype"] != np.float32 else arr
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        probs = interp.get_tensor(out["index"])[0].astype(float)

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    classes = labels.get("classes", [])
    label = classes[idx] if idx < len(classes) else "unknown"
    neg = labels.get("negative_label", "non_electronic")
    thr = float(labels.get("confidence_threshold", 0.45))
    is_electronic = (label != neg) and (conf >= thr)
    category = label.replace("_", " ")
    message = (f"Looks like {category} — photo accepted."
               if is_electronic else
               "We only accept electronics. This photo doesn't look like an electronic "
               "product — please upload a clear photo of the item you're selling.")
    return {
        "accept": is_electronic,
        "is_electronic": is_electronic,
        "model_available": True,
        "label": label,
        "category": category if is_electronic else None,
        "confidence": round(conf, 4),
        "message": message,
    }


@app.post("/detect/electronic")
async def detect_electronic(file: UploadFile = File(...)):
    """Classify an uploaded product photo — is it an electronic item?
    Used by the app at listing time to reject non-electronic photos."""
    try:
        data = await file.read()
    except Exception as e:
        return {"accept": True, "is_electronic": True, "model_available": False,
                "message": "Could not read the image.", "error": str(e)}
    if not data:
        return {"accept": True, "is_electronic": True, "model_available": False,
                "message": "Empty image."}
    try:
        return classify_electronic_image(data)
    except Exception as e:
        logger.error(f"detect_electronic failed: {e}")
        # Fail OPEN so a model error never blocks a legit listing.
        return {"accept": True, "is_electronic": True, "model_available": False,
                "message": "Image check failed; allowing upload.", "error": str(e)}


@app.get("/health")
def health():
    products = ml_models.get("products")
    return {
        "status": "online",
        "products_loaded": len(products) if products is not None else 0,
        "live_synced": ml_models.get("live_sync_count", 0),
        "refresh_in_progress": bool(ml_models.get("refresh_in_progress", False)),
        "sentiment_summary": ml_models.get("sentiment_summary") is not None,
        "sentiment_model_live": ml_models.get("sentiment_model") is not None,
        "tfidf_ready": ml_models.get("tfidf_vectorizer") is not None,
        "vocab_size": len(ml_models.get("search_vocab", [])),
        "electronic_detector": ml_models.get("electronic_interp") is not None,
    }


@app.post("/refresh")
def refresh(token: str = ""):
    """Re-sync the live catalog from the backend on demand (e.g. after a burst
    of new listings) — no redeploy needed. Protect with REFRESH_TOKEN env var.

    Runs in the background and returns immediately; watch /health
    (live_synced / refresh_in_progress) to see it complete."""
    if REFRESH_TOKEN and token != REFRESH_TOKEN:
        return {"success": False, "message": "invalid token"}
    if ml_models.get("refresh_in_progress"):
        return {"success": True, "message": "refresh already in progress"}

    ml_models["refresh_in_progress"] = True

    def _run():
        try:
            sync_live_catalog()
        except Exception as e:
            logger.error(f"On-demand refresh failed: {e}")
        finally:
            ml_models["refresh_in_progress"] = False

    threading.Thread(target=_run, daemon=True).start()
    products = ml_models.get("products")
    return {
        "success": True,
        "message": "refresh started",
        "current_products": len(products) if products is not None else 0,
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
def product_sentiment_summary(product_id: int, live: bool = True):
    # 1. Fresh live result in cache → return it.
    if live:
        with _sentiment_lock:
            cached = _sentiment_cache.get(product_id)
        if cached and (time.time() - cached[0]) < SENTIMENT_TTL:
            return cached[1]

    # 2. Classify the product's CURRENT comments with the trained model.
    #    Covers new products and new comments on old products.
    if live and ml_models.get("sentiment_model") is not None:
        live_result = classify_reviews_live(product_id)
        if live_result:
            with _sentiment_lock:
                _sentiment_cache[product_id] = (time.time(), live_result)
            return live_result

    # 3. Fall back to the baked snapshot summary.
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


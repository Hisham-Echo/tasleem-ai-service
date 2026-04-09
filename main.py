import os, pickle
import faiss, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tasleem AI Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Load pre-built data (no model downloading needed) ─────────────
index          = faiss.read_index("faiss2.index")
id_map         = pickle.load(open("id_map2.pkl",         "rb"))
reverse_id_map = pickle.load(open("reverse_id_map2.pkl", "rb"))
popular        = pickle.load(open("popular2.pkl",        "rb"))
trending       = pickle.load(open("trending2.pkl",       "rb"))
user_recs      = pickle.load(open("user2.pkl",           "rb"))
products       = pickle.load(open("products2.pkl",       "rb"))
tfidf_vectorizer, tfidf_matrix, tfidf_ids = pickle.load(open("tfidf_search.pkl", "rb"))

# ── Gemini AI (optional — free tier at aistudio.google.com) ──────
llm = None
if (key := os.getenv("GEMINI_API_KEY", "")):
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        llm = genai.GenerativeModel('gemini-1.5-flash')
        print("✓ Gemini loaded")
    except Exception as e:
        print(f"⚠ Gemini unavailable: {e}")

# ── Helpers ───────────────────────────────────────────────────────
def similar_ids(product_id: int, k: int = 10):
    """FAISS vector similarity — uses pre-stored embeddings, no model needed"""
    if product_id not in reverse_id_map:
        return []
    vec = index.reconstruct(reverse_id_map[product_id]).reshape(1, -1)
    _, I = index.search(vec, k + 1)
    return [id_map[i] for i in I[0] if id_map[i] != product_id][:k]

def search_ids(q: str, k: int = 10):
    """TF-IDF keyword search — no heavy model, ~570KB index"""
    q_vec = tfidf_vectorizer.transform([q])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    return [tfidf_ids[i] for i in top_idx if scores[i] > 0]

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "products": len(products), "users": len(user_recs)}

@app.get("/recommend/user/{user_id}")
def user_recommendations(user_id: int, last_product_id: int = None, k: int = 10):
    if user_id in user_recs and user_recs[user_id]:
        return {"section": "For You", "ids": user_recs[user_id][:k]}
    if last_product_id:
        return {"section": "Based on your last view", "ids": similar_ids(last_product_id, k)}
    return {"section": "Popular Products", "ids": popular[:k]}

@app.get("/trending")
def get_trending(k: int = 8):
    return {"section": "Trending Now", "ids": trending[:k]}

@app.get("/explore")
def get_explore(k: int = 8):
    return {"section": "Explore More", "ids": products.sample(min(k, len(products)))['id'].tolist()}

@app.get("/similar/{product_id}")
def get_similar(product_id: int, k: int = 6):
    return {"ids": similar_ids(product_id, k)}

@app.get("/search")
def search(q: str, k: int = 10):
    return {"ids": search_ids(q, k)}

@app.get("/assistant")
def assistant(query: str):
    if not llm:
        return {"answer": "AI assistant not configured. Add GEMINI_API_KEY to enable.", "ids": []}
    try:
        ids = search_ids(query, 5)
        rel = products[products['id'].isin(ids)]
        ctx = "No specific products." if rel.empty else \
              rel[['name', 'description', 'price']].to_string(index=False)
        prompt = (
            f"You are a helpful shopping assistant for Tasleem, an Egyptian marketplace. "
            f"Answer briefly (under 80 words) based on these products:\n{ctx}\n\nQuestion: {query}"
        )
        return {"answer": llm.generate_content(prompt).text, "ids": ids}
    except Exception as e:
        if "429" in str(e):
            return {"answer": "AI is busy, please try again in a moment.", "ids": []}
        return {"answer": "Sorry, couldn't process that request.", "ids": []}

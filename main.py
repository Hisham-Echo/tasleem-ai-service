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


index          = faiss.read_index("faiss2.index")
id_map         = pickle.load(open("id_map2.pkl", "rb"))
reverse_id_map = pickle.load(open("reverse_id_map2.pkl", "rb"))
popular        = pickle.load(open("popular2.pkl", "rb"))
trending       = pickle.load(open("trending2.pkl", "rb"))
user_recs      = pickle.load(open("hybrid_recs.pkl", "rb"))
products       = pickle.load(open("products2.pkl", "rb"))

tfidf_vectorizer, tfidf_matrix, tfidf_ids = pickle.load(open("tfidf_search.pkl", "rb"))

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

llm = None

try:
    api_key = os.getenv("GEMINI_API_KEY")


    if api_key:
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel('gemini-1.5-flash')
        print("✓ Gemini loaded successfully")
    else:
        print("⚠ No API key found")

except Exception as e:
    print(f"⚠ Gemini unavailable: {e}")

def similar_ids(product_id: int, k: int = 10):
    if product_id not in reverse_id_map:
        return []
    vec = index.reconstruct(reverse_id_map[product_id]).reshape(1, -1)
    _, I = index.search(vec, k + 1)
    return [id_map[i] for i in I[0] if id_map[i] != product_id][:k]


def search_ids(q: str, k: int = 10):
    """TF-IDF search"""
    q_vec = tfidf_vectorizer.transform([q])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    return [tfidf_ids[i] for i in top_idx if scores[i] > 0]


def semantic_search(query: str, k: int = 10):

    candidates = search_ids(query, k=8)

    if not candidates:
        return []

    results = []
    for pid in candidates:
        results.extend(similar_ids(pid, k=10))

    return list(dict.fromkeys(results))[:k]



@app.get("/health")
def health():
    return {"ok": True, "products": len(products), "users": len(user_recs)}


@app.get("/recommend/user/{user_id}")
def user_recommendations(user_id: int, last_product_id: int = None, k: int = 20):
    if user_id in user_recs and user_recs[user_id]:
        return {"section": "For You", "ids": user_recs[user_id][:k]}

    if last_product_id:
        return {"section": "Based on your last view", "ids": similar_ids(last_product_id, k)}

    return {"section": "Popular Products", "ids": popular[:k]}


@app.get("/trending")
def get_trending(k: int = 20):
    return {"section": "Trending Now", "ids": trending[:k]}


@app.get("/explore")
def get_explore(k: int = 20):
    return {"section": "Explore More", "ids": products.sample(min(k, len(products)))['id'].tolist()}


@app.get("/similar/{product_id}")
def get_similar(product_id: int, k: int = 20):
    return {"ids": similar_ids(product_id, k)}


@app.get("/search")
def search(q: str, k: int = 20):
    return {"ids": semantic_search(q, k)}


@app.get("/assistant")
def assistant(query: str):
    if not llm:
        return {"answer": "AI assistant not configured. Add GEMINI_API_KEY to enable.", "ids": []}

    try:
        ids = semantic_search(query, 5)

        rel = products[products['id'].isin(ids)]

        ctx = "No specific products." if rel.empty else \
              rel[['name', 'description', 'price']].to_string(index=False)

        prompt = (
            f"You are a helpful shopping assistant for Tasleem, an Egyptian marketplace. "
            f"Answer briefly (under 80 words) based on these products:\n{ctx}\n\nQuestion: {query}"
        )

        return {"answer": llm.generate_content(prompt).text, "ids": ids}


    except Exception as e:
        print("🔥 ERROR:", str(e))
        return {"answer": f"Error: {str(e)}", "ids": []}
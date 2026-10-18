import ollama
from rank_bm25 import BM25Okapi
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import CrossEncoder
import os

MODEL = 'qwen2.5:7b'
K = 10
NUM_EVAL_QUERIES = 100
DATASET = 'scifact'

print(f"Loading {DATASET} dataset...")
dataset = DATASET
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
doc_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id].get('title', '') + " " + corpus[doc_id].get('text', '') for doc_id in doc_ids]
doc_id_to_text = {doc_ids[i]: corpus_texts[i] for i in range(len(doc_ids))}

print("Building BM25 Index...")
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("Loading Cross-Encoder Reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base") # downloads the bge-reranker model from huggingface on the first run

def retrieve_top_k(query_text, k=10):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in top_indices]

def compute_rr(retrieved_docs, relevant_docs):
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0

query_ids = list(queries.keys())[:NUM_EVAL_QUERIES]

raw_hits, hybrid_hits, reranked_hits = 0, 0, 0
raw_rr, hybrid_rr, reranked_rr = [], [], []

print(f"\n--- Starting Reranker Benchmark ({NUM_EVAL_QUERIES} Queries) ---")

for i, qid in enumerate(query_ids, 1):
    raw_query = queries[qid]
    relevant_docs = set(qrels[qid].keys())
    
    # Raw BM25 (Top K)
    raw_top_k = retrieve_top_k(raw_query, k=K)
    if relevant_docs.intersection(set(raw_top_k)):
        raw_hits += 1
    raw_rr.append(compute_rr(raw_top_k, relevant_docs))
    
    # Generate LLM Expansion (Keywords + HyDE)
    kw_prompt = f"Given the English scientific search query: '{raw_query}', generate 3 to 5 synonymous English search terms or related scientific concepts. Output ONLY a space-separated list of English words, nothing else."
    kw_res = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': kw_prompt}])['message']['content'].strip()
    
    hyde_prompt = f"Write a short, factual scientific passage or abstract excerpt (2-3 sentences) in English addressing this scientific claim or query: '{raw_query}'. Do not include commentary, just the passage."
    hyde_doc = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': hyde_prompt}])['message']['content'].strip()
    
    hybrid_query = f"{raw_query} {kw_res} {hyde_doc}"
    
    # Hybrid BM25 (Top K)
    hybrid_top_k = retrieve_top_k(hybrid_query, k=K)
    if relevant_docs.intersection(set(hybrid_top_k)):
        hybrid_hits += 1
    hybrid_rr.append(compute_rr(hybrid_top_k, relevant_docs))

    # Hybrid BM25 (Top 100) -> Cross-Encoder Reranker (Top K)
    hybrid_top_100 = retrieve_top_k(hybrid_query, k=100)
    
    # Format pairs for the Cross-Encoder: [[Query, Doc1], [Query, Doc2], ...]
    pairs = [[raw_query, doc_id_to_text[doc_id]] for doc_id in hybrid_top_100]
    scores = reranker.predict(pairs)
    
    # Re-sort the top 100 documents based on the Cross-Encoder scores
    sorted_indices = np.argsort(scores)[::-1][:K]
    reranked_top_k = [hybrid_top_100[idx] for idx in sorted_indices]
    
    if relevant_docs.intersection(set(reranked_top_k)):
        reranked_hits += 1
    reranked_rr.append(compute_rr(reranked_top_k, relevant_docs))

    print(f"[{i}/{NUM_EVAL_QUERIES}] Evaluated query: {raw_query[:45]}...")

# Final Benchmark Summary
print(f"\n{'Method':<35} | {'Success@' + str(K):<12} | {'MRR@' + str(K):<10}")
print("-" * 65)
print(f"{'1. Raw BM25 (Top {K})':<35} | {raw_hits/NUM_EVAL_QUERIES:<12.2%} | {np.mean(raw_rr):<10.4f}")
print(f"{'2. Hybrid BM25 (Top {K})':<35} | {hybrid_hits/NUM_EVAL_QUERIES:<12.2%} | {np.mean(hybrid_rr):<10.4f}")
print(f"{'3. Hybrid BM25 (Top 100) + Reranker':<35} | {reranked_hits/NUM_EVAL_QUERIES:<12.2%} | {np.mean(reranked_rr):<10.4f}")
import ollama
from rank_bm25 import BM25Okapi
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

MODEL = 'qwen2.5:7b'
K = 10
NUM_EVAL_QUERIES = 30

print("Loading SciFact dataset...")
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
doc_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id].get('title', '') + " " + corpus[doc_id].get('text', '') for doc_id in doc_ids]

print("Building BM25 Index...")
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

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

raw_hits, kw_hits, hyde_hits = 0, 0, 0
raw_rr, kw_rr, hyde_rr = [], [], []

print(f"\n--- Starting 3-Way Benchmark ({NUM_EVAL_QUERIES} Queries) ---")

for i, qid in enumerate(query_ids, 1):
    raw_query = queries[qid]
    relevant_docs = set(qrels[qid].keys())
    
    # Raw BM25
    raw_top_k = retrieve_top_k(raw_query, k=K)
    if relevant_docs.intersection(set(raw_top_k)):
        raw_hits += 1
    raw_rr.append(compute_rr(raw_top_k, relevant_docs))
    
    # Keyword Expansion
    kw_prompt = (
        f"Given the English scientific search query: '{raw_query}', "
        f"generate 3 to 5 synonymous English search terms or related scientific concepts. "
        f"Output ONLY a space-separated list of English words, nothing else."
    )
    kw_res = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': kw_prompt}])['message']['content'].strip()
    kw_expanded_query = f"{raw_query} {kw_res}"
    
    kw_top_k = retrieve_top_k(kw_expanded_query, k=K)
    if relevant_docs.intersection(set(kw_top_k)):
        kw_hits += 1
    kw_rr.append(compute_rr(kw_top_k, relevant_docs))

    # HyDE Expansion
    hyde_prompt = (
        f"Write a short, factual scientific passage or abstract excerpt (2-3 sentences) in English "
        f"addressing this scientific claim or query: '{raw_query}'. Do not include commentary, just the passage."
    )
    hyde_doc = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': hyde_prompt}])['message']['content'].strip()
    hyde_query = f"{raw_query} {hyde_doc}"

    hyde_top_k = retrieve_top_k(hyde_query, k=K)
    if relevant_docs.intersection(set(hyde_top_k)):
        hyde_hits += 1
    hyde_rr.append(compute_rr(hyde_top_k, relevant_docs))

    print(f"[{i}/{NUM_EVAL_QUERIES}] Evaluated query: {raw_query[:45]}...")

# Final Benchmark Summary
print(f"\n{'Method':<20} | {'Success@' + str(K):<12} | {'MRR@' + str(K):<10}")
print("-" * 50)
print(f"{'1. Raw BM25':<20} | {raw_hits/NUM_EVAL_QUERIES:<12.2%} | {np.mean(raw_rr):<10.4f}")
print(f"{'2. Keyword Expansion':<20} | {kw_hits/NUM_EVAL_QUERIES:<12.2%} | {np.mean(kw_rr):<10.4f}")
print(f"{'3. HyDE Expansion':<20} | {hyde_hits/NUM_EVAL_QUERIES:<12.2%} | {np.mean(hyde_rr):<10.4f}")
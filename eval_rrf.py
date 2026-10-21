import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util as st_util
import os
import math
import csv
import time

K = 10
NUM_EVAL_QUERIES = 100
RRF_K = 60 # Standard smoothing constant for RRF (Reciprocal Rank Fusion)
DENSE_MODEL = 'all-MiniLM-L6-v2'
DATASETS = ['scifact', 'trec-covid', 'fiqa']
dataset = DATASETS[2]
CSV_FILE = 'baseline_rrf_benchmark_results.csv'

print(f"Loading {dataset} dataset...")
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(os.getcwd(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
doc_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id].get('title', '') + " " + corpus[doc_id].get('text', '') for doc_id in doc_ids]

print("1. Building Sparse Index (BM25)...")
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("2. Building Dense Index (Vector Embeddings)...")
dense_model = SentenceTransformer(DENSE_MODEL) 
corpus_embeddings = dense_model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

def retrieve_sparse(query_text, k=100):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in top_indices]

def retrieve_dense(query_text, k=100):
    query_embedding = dense_model.encode(query_text, convert_to_tensor=True)
    hits = st_util.semantic_search(query_embedding, corpus_embeddings, top_k=k)[0]
    return [doc_ids[hit['corpus_id']] for hit in hits]

def compute_rr(retrieved_docs, relevant_docs):
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0

def compute_ndcg(retrieved_docs, relevant_docs_with_scores, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs_with_scores:
            rel = relevant_docs_with_scores[doc_id]
            dcg += rel / math.log2(i + 2)
    ideal_docs = sorted(relevant_docs_with_scores.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_docs))
    if idcg == 0:
        return 0.0
    return dcg / idcg

query_ids = list(queries.keys())[:NUM_EVAL_QUERIES]
ACTUAL_EVAL_QUERIES = len(query_ids) # fixes math for datasets smaller than NUM_EVAL_QUERIES

sparse_hits, dense_hits, rrf_hits = 0, 0, 0
sparse_rr, dense_rr, rrf_rr = [], [], []
sparse_ndcg, dense_ndcg, rrf_ndcg = [], [], []
sparse_time_total, dense_time_total, rrf_time_total = 0.0, 0.0, 0.0

print(f"\n--- Starting Hybrid Search Benchmark ({ACTUAL_EVAL_QUERIES} Queries) ---")

for i, qid in enumerate(query_ids, 1):
    raw_query = queries[qid]
    relevant_docs_with_scores = qrels[qid] # get graded scores for nDCG
    relevant_docs = set(relevant_docs_with_scores.keys())
    
    # evaluate Sparse (BM25) (top K)
    start_time = time.time()
    sparse_top_100 = retrieve_sparse(raw_query, k=100)
    sparse_duration = time.time() - start_time
    sparse_time_total += sparse_duration
    
    sparse_top_k = sparse_top_100[:K]
    if relevant_docs.intersection(set(sparse_top_k)):
        sparse_hits += 1
    sparse_rr.append(compute_rr(sparse_top_k, relevant_docs))
    sparse_ndcg.append(compute_ndcg(sparse_top_k, relevant_docs_with_scores, k=K))
    
    # evaluate Dense (Vector Search) (top K)
    start_time = time.time()
    dense_top_100 = retrieve_dense(raw_query, k=100)
    dense_duration = time.time() - start_time
    dense_time_total += dense_duration
    
    dense_top_k = dense_top_100[:K]
    if relevant_docs.intersection(set(dense_top_k)):
        dense_hits += 1
    dense_rr.append(compute_rr(dense_top_k, relevant_docs))
    dense_ndcg.append(compute_ndcg(dense_top_k, relevant_docs_with_scores, k=K))
    
    # Reciprocal Rank Fusion (RRF)
    start_time = time.time()
    rrf_scores = {} 
    for rank, doc_id in enumerate(sparse_top_100, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank)) # Sparse score
    for rank, doc_id in enumerate(dense_top_100, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank)) # Dense score
    rrf_sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True) # sort by combined RRF score
    rrf_top_k = rrf_sorted_docs[:K]
    rrf_duration = time.time() - start_time
    rrf_time_total += (sparse_duration + dense_duration + rrf_duration)
    # evaluate RRF (Top K)
    if relevant_docs.intersection(set(rrf_top_k)):
        rrf_hits += 1
    rrf_rr.append(compute_rr(rrf_top_k, relevant_docs))
    rrf_ndcg.append(compute_ndcg(rrf_top_k, relevant_docs_with_scores, k=K))

    print(f"[{i}/{ACTUAL_EVAL_QUERIES}] Evaluated query: {raw_query[:45]}...")

# Average Latency (secs per query)
avg_sparse_time = sparse_time_total / ACTUAL_EVAL_QUERIES
avg_dense_time = dense_time_total / ACTUAL_EVAL_QUERIES
avg_rrf_time = rrf_time_total / ACTUAL_EVAL_QUERIES

# Final Benchmark Summary
print(f"\n{'Method':<35} | {'Success@' + str(K):<12} | {'MRR@' + str(K):<10} | {'nDCG@' + str(K):<10} | {'Latency (s/q)':<12}")
print("-" * 85)
print(f"{'1. Sparse BM25 (Top ' + str(K) + ')':<35} | {sparse_hits/ACTUAL_EVAL_QUERIES:<12.2%} | {np.mean(sparse_rr):<10.4f} | {np.mean(sparse_ndcg):<10.4f} | {avg_sparse_time:<12.4f}")
print(f"{'2. Dense Vector Search (Top ' + str(K) + ')':<35} | {dense_hits/ACTUAL_EVAL_QUERIES:<12.2%} | {np.mean(dense_rr):<10.4f} | {np.mean(dense_ndcg):<10.4f} | {avg_dense_time:<12.4f}")
print(f"{'3. Hybrid RRF (Top ' + str(K) + ')':<35} | {rrf_hits/ACTUAL_EVAL_QUERIES:<12.2%} | {np.mean(rrf_rr):<10.4f} | {np.mean(rrf_ndcg):<10.4f} | {avg_rrf_time:<12.4f}")

# Saving Results
file_exists = os.path.isfile(CSV_FILE)
with open(CSV_FILE, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Dataset", "Method", "Success@10", "MRR@10", "nDCG@10", "Latency(s)"])
    writer.writerow([dataset.upper(), "Sparse BM25", sparse_hits / ACTUAL_EVAL_QUERIES, np.mean(sparse_rr), np.mean(sparse_ndcg), avg_sparse_time])
    writer.writerow([dataset.upper(), "Dense Vector", dense_hits / ACTUAL_EVAL_QUERIES, np.mean(dense_rr), np.mean(dense_ndcg), avg_dense_time])
    writer.writerow([dataset.upper(), "Hybrid RRF", rrf_hits / ACTUAL_EVAL_QUERIES, np.mean(rrf_rr), np.mean(rrf_ndcg), avg_rrf_time])
print(f"\nResults securely appended to {CSV_FILE}")
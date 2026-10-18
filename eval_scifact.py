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

# load the corpus (documents) and queries
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
doc_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id].get('title', '') + " " + corpus[doc_id].get('text', '') for doc_id in doc_ids]
print(f"Successfully loaded {len(corpus_texts)} documents and {len(queries)} test queries.")

# build the BM25 Index
print("Building BM25 Index...")
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# testing first query
query_id = list(queries.keys())[0]
raw_query = queries[query_id]

# expanding with the LLM
prompt = (
        f"Given the English scientific search query: '{raw_query}', "
        f"generate 3 to 5 synonymous English search terms or related scientific concepts. "
        f"Output ONLY a space-separated list of English words, nothing else."
    )
print(f"\nQuerying {MODEL} for expansion on : '{raw_query}'...")
response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
expanded_terms = response['message']['content'].strip()

expanded_query = f"{raw_query} {expanded_terms}"

print(f"\nRaw Query : {raw_query}")
print(f"Expanded Query : {expanded_query}")

# evaluation setup (searches BM25 and returns the top k document IDs)

def retrieve_top_k(query_text, k=10):
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in top_indices]

def compute_rr(retrieved_docs, relevant_docs): # computes reciprocal rank for a single query
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0

print(f"\n--- Starting Evaluation ({NUM_EVAL_QUERIES} Queries) ---")

query_ids = list(queries.keys())[:NUM_EVAL_QUERIES]  # fetching first NUM_EVAL_QUERIES
raw_hits, expanded_hits = 0, 0
raw_rr_list, expanded_rr_list = [], []

for i, qid in enumerate(query_ids, 1):
    raw_query = queries[qid]
    relevant_docs = set(qrels[qid].keys()) # fetch the doc IDs that are relevant to this query
    
    # raw query retrieval
    raw_top_k = set(retrieve_top_k(raw_query, k=K))
    if relevant_docs.intersection(raw_top_k):
        raw_hits += 1
    raw_rr_list.append(compute_rr(raw_top_k, relevant_docs))
        
    # expanded query retrieval (forced english)
    prompt = (
        f"Given the English scientific search query: '{raw_query}', "
        f"generate 3 to 5 synonymous English search terms or related scientific concepts. "
        f"Output ONLY a space-separated list of English words, nothing else."
    )
    response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
    expanded_terms = response['message']['content'].strip()
    expanded_query = f"{raw_query} {expanded_terms}"
    
    expanded_top_k = set(retrieve_top_k(expanded_query, k=K))
    if relevant_docs.intersection(expanded_top_k):
        expanded_hits += 1
    expanded_rr_list.append(compute_rr(expanded_top_k, relevant_docs))
    
    print(f"[{i}/{NUM_EVAL_QUERIES}] Evaluated : {raw_query[:40]}...")

# results
print(f"\n--- Final Results ({NUM_EVAL_QUERIES} Queries) ---")
print(f"Raw Success@{K}      : {raw_hits}/{NUM_EVAL_QUERIES} ({raw_hits/NUM_EVAL_QUERIES:.2%})")
print(f"Expanded Success@{K} : {expanded_hits}/{NUM_EVAL_QUERIES} ({expanded_hits/NUM_EVAL_QUERIES:.2%})")
print(f"Raw MRR@{K}          : {np.mean(raw_rr_list):.4f}")
print(f"Expanded MRR@{K}     : {np.mean(expanded_rr_list):.4f}")
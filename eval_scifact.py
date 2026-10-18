import ollama
from rank_bm25 import BM25Okapi
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

MODEL = 'qwen2.5:7b'

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

# building the BM25 Index
print("Building BM25 Index...")
tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

# testing first query
query_id = list(queries.keys())[0]
raw_query = queries[query_id]

# expanding with the LLM
prompt = f"Given the search query: '{raw_query}', generate 3 to 5 synonymous search terms. Output ONLY a space-separated list of words."
print(f"\nQuerying {MODEL} for expansion on: '{raw_query}'...")
response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
expanded_terms = response['message']['content'].strip()

expanded_query = f"{raw_query} {expanded_terms}"

print(f"\nRaw Query : {raw_query}")
print(f"Expanded Query : {expanded_query}")
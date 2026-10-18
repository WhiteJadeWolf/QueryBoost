import ollama
from rank_bm25 import BM25Okapi
import numpy as np

MODEL = 'qwen2.5:7b'
corpus = [
    "Deep learning models are used to detect fake news and disinformation online.",
    "Baking sourdough bread requires flour, water, salt, and wild yeast.",
    "Natural language processing helps detect misinformation across social media platforms."
]
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# testing model connection
raw_query = "AI detecting fake news"
prompt = f"Given the search query: '{raw_query}', generate 3 to 5 synonymous search terms. Output ONLY a space-separated list of words."

print("Waiting for Qwen2.5:7b...")
response = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
expanded_terms = response['message']['content'].strip()

expanded_query = f"{raw_query} {expanded_terms}"

print(f"\nRaw Query : {raw_query}")
print(f"Expanded Query : {expanded_query}")
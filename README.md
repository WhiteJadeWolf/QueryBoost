# QueryBoost

An Information Retrieval (IR) evaluation pipeline that compares traditional BM25 search against Large Language Model (LLM) query expansion techniques.

This project uses `qwen2.5:7b` (via Ollama) to dynamically expand search queries and evaluates the performance on the **SciFact** dataset.

## Benchmark Results

Evaluated on 100 scientific queries from the SciFact dataset (Top-10 Retrieval):

| Method                     | Success@10 | MRR@10     |
| -------------------------- | ---------- | ---------- |
| **1. Raw BM25 (Baseline)** | 79.00%     | 0.5909     |
| **2. Keyword Expansion**   | 82.00%     | 0.6633     |
| **3. HyDE Expansion**      | **86.00%** | 0.6632     |
| **4. Hybrid (KW + HyDE)**  | 85.00%     | **0.6938** |

### Key Findings

- **HyDE** (Hypothetical Document Embeddings) yielded the highest recall (Success@10), proving highly effective at capturing semantic context.
- **Hybrid Expansion** (Keywords + HyDE) yielded the highest precision (MRR@10), meaning it ranks the correct documents higher on average than any single method.

## Tech Stack

- **Retriever:** `rank-bm25` (BM25Okapi)
- **LLM Engine:** Local `qwen2.5:7b` via Ollama
- **Datasets & Evaluation:** `beir`

# QueryBoost : Information Retrieval Pipeline

QueryBoost is a modular benchmarking engine designed to evaluate, compare, and visualize the trade-offs between traditional fast retrieval methods and modern, computationally heavy generative search pipelines with the help of LLMs. 

This project tests various Information Retrieval (IR) architectures across standard BEIR datasets (SciFact, TREC-COVID, FIQA) to quantify the central trade-off of modern search engines : absolute semantic accuracy versus operational latency.

## Core Technologies and Retrieval Methods


### 1. BM25 (Sparse Lexical Search)
* **Role:** The foundational exact-keyword matching baseline. It counts search term frequency within documents, weighting rare words higher than common ones.
* **Location in Results:** Evaluated in both pipelines. It appears as "Sparse BM25" in `eval_rrf.py` and as "Raw BM25" in `eval_reranker.py`.
* **Strengths:** Extremely fast, computationally cheap, and highly effective for specific names, IDs, or domain jargon.
* **Weaknesses:** Suffers from "vocabulary mismatch." If a user searches for "automobile" but the document uses "car," BM25 will completely miss the document.

### 2. Dense Vector Search (Semantic Search)
* **Role:** Uses a neural network (`all-MiniLM-L6-v2`) to convert queries and documents into mathematical vectors, grouping documents with similar meanings close together.
* **Location in Results:** Evaluated strictly in `eval_rrf.py` as the highly efficient semantic baseline.
* **Strengths:** Understands context and synonyms. It can match concepts even if exact letters do not overlap.
* **Weaknesses:** Can drift away from a user's exact intent, sometimes missing documents containing a highly specific, rare keyword if the general semantic "vibe" does not align.

### 3. RRF (Reciprocal Rank Fusion)
* **Role:** The mathematical function that merges the Top 100 BM25 list and the Top 100 Dense Vector list. Since raw lexical and vector scores exist on different scales, RRF combines them based on rank positions using the formula: `1 / (60 + rank)`.
* **Location in Results:** The ultimate output of `eval_rrf.py`, plotted as "Hybrid RRF".
* **Strengths:** Rewards documents that perform well in both lexical and semantic systems, producing the most balanced baseline metrics.

### 4. HyDE (Hypothetical Document Embeddings) & LLM Expansion
* **Role:** A query expansion technique utilizing a local Large Language Model (`qwen2.5:7b` via Ollama) to write a hypothetical, realistic-sounding passage answering the user's short query. This fake answer is then vectorized to search the real database.
* **Location in Results:** The intermediate step in `eval_reranker.py`, plotted as "Hybrid BM25".
* **Strengths:** Solves the "asymmetry" problem of search by bridging short queries with long database documents, giving the Vector Search a perfect semantic template.
* **Weaknesses:** Introduces significant latency. Furthermore, if the LLM hallucinates a completely incorrect answer, it drags the search results in the wrong direction.

### 5. Cross-Encoder Reranker
* **Role:** A heavy, deep-learning model (`BAAI/bge-reranker-base`) that performs bidirectional cross-attention scoring. Unlike Dense Vectors, a Cross-Encoder feeds the query and document into the neural network simultaneously to map every word's relationship to one another.
* **Location in Results:** The final output of `eval_reranker.py`, plotted as "Hybrid + Reranker".
* **Strengths:** Produces the absolute highest quality, most accurate relevance scores.
* **Weaknesses:** Computationally massive. It cannot be run across a full database, which is why it is used purely as a "Second-Stage" step to re-order the top 100 candidates grabbed by earlier methods.

## Architecture & Performance Trade-offs

* **Raw BM25 & Dense Vectors (The True Baseline):** These operate in fractions of a second, averaging 0.01 to 0.5 seconds per query. This is ideal for instant consumer search expectations.

* **Hybrid BM25 (LLM Expansion):** Generating the HyDE passage and keywords through Ollama takes roughly 1.3 to 4.0 seconds per query. This attempts to lift the lexical baseline by providing more text hooks, but pushes response times beyond standard real-time expectations.

* **Hybrid + Reranker:** This represents the massive bottleneck. Evaluating 100 separate document pairs sequentially against a deep cross-attention neural network balloons the latency to 26 to 35 seconds per query. While it achieves the highest nDCG@10 scores and overcomes domain-specific jargon mismatches (like in the FIQA dataset), it is generally unsuitable for real-time applications without dedicated hardware acceleration.

To demonstrate these differences interactively, the Streamlit frontend features a toggle switch between a "Fast Search" mode (Baseline RRF) and a "Deep Semantic Search" mode (Hybrid + Reranker).

## Project Structure


```
├── 📁 results                  # Generated CSV metrics and 2x2 grid visualizations
│   ├── 📄 sample.csv
│   └── 🖼️ sample.png
├── 📝 README.md                # Project documentation
├── 🐍 app.py                   # Streamlit interactive frontend and analytics dashboard
├── 🐍 eval_hyde.py             # Evaluation script isolating LLM Query Expansion (HyDE) performance
├── 🐍 eval_reranker.py         # Evaluation script for LLM Query Expansion and Cross-Encoder (deep generative pipeline)
├── 🐍 eval_rrf.py              # Evaluation script for BM25, Dense Vector, and RRF (fast baseline pipeline)
├── 🐍 eval_scifact.py          # Dataset-specific script for evaluating retrieval on the SciFact corpus
├── 📄 requirements.txt         # Project dependencies
├── 🐍 test.py                  # Sandbox script for testing individual functions, API calls, or models
└── 🐍 visualize_results.py     # Matplotlib/Seaborn visualization generator
``` 


## Installation

1. **Clone the repository :**
     ```
     git clone https://github.com/WhiteJadeWolf/QueryBoost.git
     cd QueryBoost
     ```

2. **Create and activate a virtual environment :**
     ```
     conda create -n <env_name> python=3.10
     conda activate <env_name>
     ```

3. **Install Project dependencies :**
     ```
     pip install -r requirements.txt
     ```
> *Note : Ensure you have `torch` and `torchvision` installed according to your specific hardware setup.*

4. **Install and run Ollama :**

The generative pipeline requires a local instance of Ollama running a local LLM (default = `qwen2.5:7b`).
* Download [Ollama](https://ollama.com/)

* Pull the model :
     ```
     ollama run <model_name>
     ```
     By default :
     ```
     ollama run qwen2.5:7b
     ```
> *NOTE ON CUSTOM LLM MODELS : By default, the generative pipeline uses the `qwen2.5:7b` model via Ollama. If you are using a different model (e.g., `llama3`, `mistral`, or `phi3`), you must update the model parameter in the Python scripts. Search the code for `qwen2.5:7b` (located in the `MODEL` constant in `app.py` and the `ollama.chat()` functions in `eval_reranker.py` and `eval_hyde.py`) and replace it with your downloaded model's exact name before running the evaluations or frontend.*

## Usage

### 1. Run the Streamlit Application
The easiest way to interact with the project is through the frontend UI. The app will automatically load the models into memory and provide an interactive search bar alongside the benchmarking dashboards.
```
streamlit run app.py
```

### 2. Run Benchmarks via Command Line
To generate fresh benchmark data across 100 queries for a specific BEIR dataset, run the evaluation scripts directly:

#### *Run the fast baseline evaluation (BM25, Dense, Hybrid RRF) :*
```
python eval_rrf.py
```

#### *Run the deep semantic evaluation (HyDE + Cross-Encoder) :*
```
python eval_reranker.py
```

> *Note : You can change the target BEIR dataset (e.g., "scifact", "trec-covid", "fiqa") by modifying the `dataset` variable at the top of the evaluation scripts.*

### 3. Generate Visualizations
Once the evaluations are complete, you can generate 2x2 grid visualizations of the metrics. These will be saved directly to the `results/` folder.
```
python visualize_results.py baseline_rrf_benchmark_results.csv
python visualize_results.py query_exp_benchmark_results.csv
```

## Tech Stack

* **Language :** Python 3.10

* **Frontend :** Streamlit

* **Models & Inference :** 
  * SentenceTransformers (`all-MiniLM-L6-v2`, `BAAI/bge-reranker-base`)
  * Ollama (`qwen2.5:7b`)

* **Retrieval & Datasets :** `rank_bm25`, BEIR

* **Data & Visualization :** Pandas, Matplotlib, Seaborn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import ollama
import numpy as np

RESULTS_DIR = 'results'
MODEL = 'qwen2.5:7b'

# Page Configuration
st.set_page_config(
    page_title="QueryBoost IR Engine",
    page_icon="👌🏾",
    layout="wide"
)

# caching models and datasets so they don't reload on every button click.
@st.cache_resource(show_spinner="Loading BEIR Dataset & Models...")
def load_backend():
    from eval_rrf import retrieve_sparse, retrieve_dense, corpus, doc_ids # standard baseline functions
    from eval_reranker import reranker, retrieve_top_k # reranker from LLM generative script
    return {
        "retrieve_sparse": retrieve_sparse,
        "retrieve_dense": retrieve_dense,
        "retrieve_top_k": retrieve_top_k, 
        "reranker": reranker,
        "corpus": corpus,
        "doc_ids": doc_ids
    }

backend = load_backend()

# Main UI Header
st.title("QueryBoost : Information Retrieval Pipeline")
st.markdown("A modular benchmarking engine testing the trade-offs between speed and semantic depth.")

tab_search, tab_analytics = st.tabs(["Interactive Search Demo", "Benchmark Analytics Dashboard"]) # Tab Layout

# TAB 1 : INTERACTIVE SEARCH DEMO
with tab_search:
    st.header("Test the Retrieval Pipelines")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("Enter a search query:", placeholder="e.g., Effects of caffeine on deep sleep...")
    with col2:
        pipeline_mode = st.selectbox("Select Pipeline Engine :", [
            "Fast : Baseline RRF", 
            "Deep : Generative LLM + Reranker"
        ])

    if st.button("Search", type="primary") and user_query:
        start_time = time.time()
        
        if "Fast" in pipeline_mode:
            
            with st.spinner("Running Sparse & Dense Vectors... fusing ranks..."):
                sparse_results = backend["retrieve_sparse"](user_query, k=100)
                dense_results = backend["retrieve_dense"](user_query, k=100)
                rrf_scores = {}
                for rank, doc_id in enumerate(sparse_results, 1):
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
                for rank, doc_id in enumerate(dense_results, 1):
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
                rrf_top_10 = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]
                latency = time.time() - start_time
                st.success(f"⚡ Search completed in {latency:.4f} seconds")
                st.subheader("Top Results (Hybrid RRF)")
                for i, doc_id in enumerate(rrf_top_10, 1):
                    doc_data = backend["corpus"][doc_id]
                    doc_title = doc_data.get('title', 'No Title')
                    doc_text = doc_data.get('text', '')
                    doc_score = rrf_scores[doc_id]
                    with st.expander(f"Result {i} | {doc_title} [RRF Score: {doc_score:.4f}]"):
                        st.write(doc_text)
                        

        elif "Deep" in pipeline_mode:
            
            with st.spinner(f"Expanding query via Ollama ({MODEL})..."):
                kw_prompt = f"Given the search query: '{user_query}', generate 3 to 5 synonymous English search terms or related concepts. Output ONLY a space-separated list of English words, nothing else."
                kw_res = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': kw_prompt}])['message']['content'].strip()
                hyde_prompt = f"Write a short, factual passage (2-3 sentences) in English directly addressing or answering this query: '{user_query}'. Do not include commentary, just the passage."
                hyde_doc = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': hyde_prompt}])['message']['content'].strip()
                st.info(f"**Generated Keywords:** {kw_res}")
                st.info(f"**HyDE Passage:** {hyde_doc}")
                hybrid_query = f"{user_query} {kw_res} {hyde_doc}"
                
            with st.spinner("Retrieving Top 100 & Scoring via Cross-Encoder..."):
                hybrid_top_100 = backend["retrieve_top_k"](hybrid_query, k=100)
                pairs = [[user_query, backend["corpus"][doc_id].get('title', '') + " " + backend["corpus"][doc_id].get('text', '')] for doc_id in hybrid_top_100]
                scores = backend["reranker"].predict(pairs)
                sorted_indices = np.argsort(scores)[::-1][:10] # Sort by reranker score
                latency = time.time() - start_time
                st.warning(f"Deep Semantic Search completed in {latency:.4f} seconds")
                st.subheader("Top Results (Cross-Encoder Reranked)")
                for i, idx in enumerate(sorted_indices, 1):
                    doc_id = hybrid_top_100[idx]
                    doc_data = backend["corpus"][doc_id]
                    doc_title = doc_data.get('title', 'No Title')
                    doc_text = doc_data.get('text', '')
                    doc_score = scores[idx]
                    with st.expander(f"Result {i} | {doc_title} [Cross-Attention Score: {doc_score:.4f}]"):
                        st.write(doc_text)

# TAB 2 : BENCHMARK ANALYTICS DASHBOARD
with tab_analytics:
    st.header("Pipeline Performance Metrics")
    st.markdown(f"Visualizing empirical data from the `{RESULTS_DIR}/` directory.")
    
    # Select CSV to visualize
    available_csvs = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")] if os.path.exists(RESULTS_DIR) else []
    
    if not available_csvs:
        st.error("No CSV files found in the 'results/' directory. Please run your evaluation scripts first.")
    else:
        selected_csv = st.selectbox("Select Benchmark Data :", available_csvs)
        csv_path = os.path.join(RESULTS_DIR, selected_csv)
        df = pd.read_csv(csv_path)
        with st.expander("View Raw Data Table"): # Display raw data table
            st.dataframe(df, width='stretch')
            
        # Render plots dynamically
        st.subheader("Comprehensive Metric Analysis")
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        pipeline_name = "Baseline RRF" if "rrf" in selected_csv.lower() else "Generative LLM (HyDE + Reranker)"
        fig.suptitle(f"{pipeline_name} Pipeline - Comprehensive Metrics", fontsize=20, fontweight='bold', y=0.98)

        metrics = [
            {"col": "Success@10", "ax": axes[0, 0], "title": "Success@10 (Recall/Hit Rate)", "fmt": ".2%"},
            {"col": "MRR@10", "ax": axes[0, 1], "title": "Mean Reciprocal Rank (MRR@10)", "fmt": ".4f"},
            {"col": "nDCG@10", "ax": axes[1, 0], "title": "nDCG@10", "fmt": ".4f"},
            {"col": "Latency(s)", "ax": axes[1, 1], "title": "Average Latency (Seconds/Query)", "fmt": ".4f"}
        ]

        # Loop through and plot
        for m in metrics:
            ax = m["ax"]
            if m["col"] in df.columns:
                sns.barplot(data=df, x="Dataset", y=m["col"], hue="Method", palette="viridis", ax=ax, edgecolor="black")
                ax.set_title(m["title"], fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel("BEIR Dataset", fontsize=12)
                ax.set_ylabel("Score" if "Latency" not in m["col"] else "Time (Seconds)", fontsize=12)
                max_val = df[m["col"]].max()
                ax.set_ylim(0, max_val + (max_val * 0.2))
                for p in ax.patches:
                    height = p.get_height()
                    if pd.notna(height) and height > 0:
                        label_text = f"{height:{m['fmt']}}"
                        ax.annotate(label_text, (p.get_x() + p.get_width() / 2., height), 
                                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=10)
                ax.get_legend().remove()
            else:
                ax.text(0.5, 0.5, f"Column {m['col']} not found", ha='center')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Retrieval Method", bbox_to_anchor=(1.02, 0.95), loc='upper left', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Pass the matplotlib figure directly to Streamlit
        st.pyplot(fig)
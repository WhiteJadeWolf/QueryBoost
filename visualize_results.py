import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = 'results'

sns.set_theme(style="whitegrid")

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "query_exp_benchmark_results.csv"
csv_file = os.path.join(RESULTS_DIR, filename)

if not os.path.exists(csv_file):
    print(f"Error : '{csv_file}' not found.")
    print("Please run your evaluation scripts first to generate the benchmark data.")
    sys.exit(1)
    
print(f"Loading data from : {csv_file}")
df = pd.read_csv(csv_file)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

pipeline_name = "Baseline RRF" if "rrf" in filename.lower() else "Generative LLM (HyDE + Reranker)"
fig.suptitle(f"QueryBoost : {pipeline_name} Pipeline - Comprehensive Metrics", fontsize=20, fontweight='bold', y=0.98)

metrics = [
    {"col": "Success@10", "ax": axes[0, 0], "title": "Success@10 (Recall/Hit Rate)", "fmt": ".2%"},
    {"col": "MRR@10", "ax": axes[0, 1], "title": "Mean Reciprocal Rank (MRR@10)", "fmt": ".4f"},
    {"col": "nDCG@10", "ax": axes[1, 0], "title": "nDCG@10", "fmt": ".4f"},
    {"col": "Latency(s)", "ax": axes[1, 1], "title": "Average Latency (Seconds/Query)", "fmt": ".4f"}
]

for m in metrics:
    ax = m["ax"]
    sns.barplot(
        data=df, 
        x="Dataset", 
        y=m["col"], 
        hue="Method", 
        palette="viridis", 
        ax=ax,
        edgecolor="black"
    )
    
    ax.set_title(m["title"], fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("BEIR Dataset", fontsize=12)
    ax.set_ylabel("Score" if "Latency" not in m["col"] else "Time (Seconds)", fontsize=12)
    
    max_val = df[m["col"]].max()
    ax.set_ylim(0, max_val + (max_val * 0.2))
    
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height) and height > 0:
            label_text = f"{height:{m['fmt']}}"
            ax.annotate(label_text, 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=10)
    ax.get_legend().remove()

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title="Retrieval Method", bbox_to_anchor=(1.02, 0.95), loc='upper left', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_filename = csv_file.replace(".csv", "_plot.png")
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved plot to {output_filename}")
plt.show()
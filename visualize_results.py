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
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(
    data=df, 
    x="Dataset", 
    y="nDCG@10", 
    hue="Method", 
    palette="viridis", 
    ax=ax,
    edgecolor="black"
)

pipeline_name = "Baseline RRF" if "rrf" in csv_file.lower() else "Generative LLM (HyDE + Reranker)"
ax.set_title(f"IR Project : {pipeline_name} Performance Across Domains", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel("nDCG@10 Score", fontsize=12, fontweight='bold')
ax.set_xlabel("BEIR Dataset", fontsize=12, fontweight='bold')
max_score = df["nDCG@10"].max()
ax.set_ylim(0, max_score + (max_score * 0.2))

for p in ax.patches:
    height = p.get_height()
    if pd.notna(height) and height > 0:
        ax.annotate(f'{height:.4f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=10)

plt.legend(title="Retrieval Method", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_filename = csv_file.replace(".csv", "_plot.png")
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved plot to {output_filename}")
plt.show()
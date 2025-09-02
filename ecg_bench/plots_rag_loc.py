# Grouped bars by RAG Location (only Training✓ & Inference✓ rows)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# Bigger fonts globally
mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# X categories and methods
locations = ["System prompt", "User query"]
methods = ["RAG (Retrieved Full)", "RAG (Retrieved Report)"]
colors  = {"RAG (Retrieved Full)": "tab:orange", "RAG (Retrieved Report)": "tab:blue"}

# Data: means and SDs for the ✓/✓ rows only, ordered as [System prompt, User query]
data = {
    "Bleu-4": {
        "RAG (Retrieved Full)":  {"mean": [36.24,  3.82], "sd": [0.05, 0.04]},
        "RAG (Retrieved Report)": {"mean": [38.08, 38.03], "sd": [0.10, 0.03]},
    },
    "Rouge-L": {
        "RAG (Retrieved Full)":  {"mean": [72.67, 29.49], "sd": [0.01, 0.07]},
        "RAG (Retrieved Report)": {"mean": [75.61, 75.81], "sd": [0.04, 0.18]},
    },
    "Meteor": {
        "RAG (Retrieved Full)":  {"mean": [68.64, 34.54], "sd": [0.09, 0.14]},
        "RAG (Retrieved Report)": {"mean": [69.85, 70.00], "sd": [0.08, 0.02]},
    },
    "BertScore F1": {
        "RAG (Retrieved Full)":  {"mean": [96.89, 87.80], "sd": [0.00, 0.01]},
        "RAG (Retrieved Report)": {"mean": [97.49, 97.52], "sd": [0.01, 0.01]},
    },
    "Accuracy": {
        "RAG (Retrieved Full)":  {"mean": [16.66,  0.15], "sd": [0.12, 0.05]},
        "RAG (Retrieved Report)": {"mean": [18.11, 18.17], "sd": [0.10, 0.23]},
    },
}

# Optional tidy DataFrame for inspection/export
rows = []
for metric, md in data.items():
    for method in methods:
        for li, loc in enumerate(locations):
            rows.append({
                "RAG Location": loc,
                "Method": method,
                "Metric": metric,
                "Mean": md[method]["mean"][li],
                "SD": md[method]["sd"][li],
            })
df = pd.DataFrame(rows)

def grouped_bar(metric_name, filename):
    md = data[metric_name]
    means_full   = md[methods[0]]["mean"];  sds_full   = md[methods[0]]["sd"]
    means_report = md[methods[1]]["mean"];  sds_report = md[methods[1]]["sd"]

    x = np.arange(len(locations)); width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, means_full,   width, yerr=sds_full,   capsize=4,
           label=methods[0])
    ax.bar(x + width/2, means_report, width, yerr=sds_report, capsize=4,
           label=methods[1])

    ax.set_xticks(x)
    ax.set_xticklabels(locations)
    ax.set_ylabel(metric_name)

    # Legend outside (right side)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(f"./{filename}", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Generate one chart per metric
grouped_bar("Bleu-4", "bleu4_vs_ragloc.png")
grouped_bar("Rouge-L", "rougel_vs_ragloc.png")
grouped_bar("Meteor", "meteor_vs_ragloc.png")
grouped_bar("BertScore F1", "bertscore_vs_ragloc.png")
grouped_bar("Accuracy", "accuracy_vs_ragloc.png")

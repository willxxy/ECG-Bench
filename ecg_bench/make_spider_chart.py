import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data setup
metrics = ['Bleu-4', 'Rouge-L', 'Meteor', 'BertScore F1', 'Accuracy']
data = {
    'MERL (X_sig)': [19.77, 37.09, 25.69, 95.06, 24.66],
    'ST-MEM (X_sig)': [16.69, 33.67, 22.97, 95.15, 21.68],
    'MLAE (X_sig)': [19.70, 34.62, 24.43, 95.08, 21.07],
    'MTAE (X_sig)': [15.00, 37.44, 24.23, 95.01, 26.55],
    'CLIP (X*_sig)': [12.84, 35.02, 22.89, 94.56, 24.42],
    'ViT (X*_sig)': [16.44, 35.61, 23.95, 94.84, 23.87],
    'SigLIP (X*_sig)': [13.02, 33.26, 22.68, 94.71, 22.45],
    'CLIP (X_img)': [19.92, 37.82, 25.30, 95.15, 25.38],
    'ViT (X_img)': [9.99, 36.50, 22.42, 95.12, 27.20],
    'SigLIP (X_img)': [14.55, 29.86, 20.21, 95.50, 19.27],
    'ECG-Byte (X_id)': [16.47, 39.05, 24.94, 95.35, 27.18],
}

df = pd.DataFrame(data, index=metrics)

# Global (per-metric) normalization across all models
df_norm = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1) - df.min(axis=1), axis=0)

# Angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

# Plot setup
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(df_norm.columns))]

# Replace ECG-Byte's color with black
ecg_byte_idx = list(df_norm.columns).index('ECG-Byte (X_id)')
colors[ecg_byte_idx] = 'black'

# Plot each model
for idx, model in enumerate(df_norm.columns):
    values = df_norm[model].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model, color=colors[idx], linewidth=2)
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

# Configure fonts and layout
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=16, fontweight='bold')
ax.set_title('ECG-QA MIMIC-IV-ECG', size=20, pad=20, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)
ax.grid(color='gray', linestyle='--', linewidth=0.5)

# Legend outside with bold font
# ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), fontsize=14, prop={'weight': 'bold'})

# Save outputs
fig.savefig('./spider_chart_updated.png', dpi=300, bbox_inches='tight')
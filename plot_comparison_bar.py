import matplotlib.pyplot as plt
import numpy as np

# Data provided by the user
models = ['CLAE (ours)', 'WavLM', 'Whisper-tiny', 'ECAPA', 'emotion2vec', 'Mimi']
macro_f1 = [49.6, 60.7, 63.2, 52.3, 63.0, 61.6]
macro_f1_err = [6.2, 8.0, 5.8, 4.9, 6.4, 7.7]

accuracy = [50.0, 61.0, 63.3, 52.5, 63.3, 62.3]
accuracy_err = [6.6, 8.2, 5.9, 5.1, 6.7, 7.3]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, macro_f1, width, yerr=macro_f1_err, label='Macro-F1', color='#1f77b4', capsize=5, alpha=0.8)
rects2 = ax.bar(x + width/2, accuracy, width, yerr=accuracy_err, label='Accuracy', color='#ff7f0e', capsize=5, alpha=0.8)

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Performance Comparison on Speech Emotion Recognition', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 20),  # 20 points vertical offset to avoid overlap with error bars
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/model_comparison_bar.png', dpi=300)
print("Bar chart generated successfully.")

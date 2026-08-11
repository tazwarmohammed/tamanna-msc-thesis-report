import matplotlib.pyplot as plt
import numpy as np

models = ['CLAE (ours)', 'WavLM', 'Whisper-tiny', 'ECAPA', 'emotion2vec', 'Mimi', 'Higgs V2']
accuracy = [65.6, 29.6, 35.0, 95.2, 32.3, 64.7, 73.7]

x = np.arange(len(models))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 6))
rects = ax.bar(x, accuracy, width, color='#2ca02c', alpha=0.8)

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Performance on Closed-Set Speaker Identification', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects)
fig.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/speaker_id_bar.png', dpi=300)
print("Speaker ID Bar chart generated successfully.")

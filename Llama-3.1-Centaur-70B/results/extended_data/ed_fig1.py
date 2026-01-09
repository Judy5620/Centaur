import torch
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
import matplotlib.gridspec as gridspec
from datasets import load_dataset
import numpy as np
from transformers import pipeline
from sklearn.manifold import MDS

gs = gridspec.GridSpec(2, 2, width_ratios=[0.5, 0.5])

Bs = torch.load('../contamination/results/Bs.pth')

log_Bs = torch.log(Bs)

fig = plt.figure(figsize=(7.20472, 4))
plt.style.use(['nature'])

ax = fig.add_subplot(gs[0, 0])

image = plt.imread('overview.png')
cax = ax.imshow(image)
ax.axis('off')

ax = fig.add_subplot(gs[0, 1])

image = plt.imread('wordcloud.png')
cax = ax.imshow(image)
ax.axis('off')

ax1 = fig.add_subplot(gs[1, 0])

ax1.scatter(torch.arange(len(log_Bs)), log_Bs, color='#69005f')
ax1.axhline(y=1, color='grey', linestyle='--', linewidth=1.0)

ax1.text(len(log_Bs), 1.07, 'potentially contaminated', fontsize=6, color='red', horizontalalignment='right')
ax1.text(len(log_Bs), 0.83, 'not contaminated', fontsize=6, color='green', horizontalalignment='right')
ax1.set_ylabel(r'$\log B$')
ax1.set_xlabel('Experiment')
ax1.set_ylim(-1.6, 1.1)
ax1.set_xlim(-0.5, len(log_Bs)+0.1)

ax2 = fig.add_subplot(gs[1, 1])

eval_experiments_names = [
    'Modified problem structure (Figure 3b)',
    'Modified cover story (Figure 3a)',
    'Entirely novel domain (Figure 3c)',
    'Moral decision-making',
    'Naturalistic category learning',
    'Naturalistic reward learning',
    'Economic games',
    'Behavioral propensities',
    'Deep sequential decision task',
]


embeddings = np.load('embeddings.npy')
colors = 76 * ['#69005f']
colors.extend(['C0', 'C1', 'C2', '#ff506e', '#ff506e', '#ff506e', '#ff506e', '#ff506e', '#ff506e'])

ax2.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=25, alpha=0.8)
ax2.set_xlabel('Embedding dimension 1')
ax2.set_ylabel('Embedding dimension 2')

for i in range(1, len(eval_experiments_names) + 1):
    plt.annotate(eval_experiments_names[-i], (0.5 + embeddings[-i, 0], embeddings[-i, 1]-0.2), size=5)

fig.text(0.015, 0.955, 'a', fontsize=8, weight='bold')
fig.text(0.015, 0.52, 'c', fontsize=8, weight='bold')
fig.text(0.478, 0.955, 'b', fontsize=8, weight='bold')
fig.text(0.478, 0.52, 'd', fontsize=8, weight='bold')

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig1.jpg', bbox_inches='tight', dpi=300)
plt.show()

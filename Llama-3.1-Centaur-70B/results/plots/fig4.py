import matplotlib.pyplot as plt
import torch
import seaborn as sns
import scienceplots
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import glob
from natsort import natsorted
import pandas as pd
import numpy as np
from sklearn.manifold import MDS

gs = gridspec.GridSpec(1, 3, width_ratios=[0.33333, 0.33333, 0.33333])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 7.08661/3))

# plot MDS
metrics_df = pd.read_csv('../results/CogBench/behaviour.csv')
metrics_df = metrics_df.loc[:, ~metrics_df.columns.str.contains('^Unnamed')]
colors = ['black' for _ in metrics_df.Agent]
colors = ['grey' if engine in ['Human'] else color for engine, color in zip(metrics_df.Agent, colors)]
colors = ['#69005f' if engine == 'Centaur' else color for engine, color in zip(metrics_df.Agent, colors)]
colors = ['#ff506e' if engine == 'Llama' else color for engine, color in zip(metrics_df.Agent, colors)]

reducer = MDS(n_components=2, random_state=1)
metrics_scores = metrics_df.iloc[:, 1:metrics_df.shape[1]//2].values
agent_names = metrics_df.iloc[:, 0].values
embedding = reducer.fit_transform(metrics_scores)

ax = fig.add_subplot(gs[:, 0])
ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=25, alpha=0.8)
ax.set_xlabel('Embedding dimension 1')
ax.set_ylabel('Embedding dimension 2')
ax.set_xlim(-4, 6)
ax.set_ylim(-4, 6)

for i in range(embedding.shape[0]):
    if agent_names[i] == 'GPT-3.5':
        ax.annotate(agent_names[i], (-0.5 + embedding[i, 0], embedding[i, 1]+0.5))
    else:
        ax.annotate(agent_names[i], (0.4 + embedding[i, 0], embedding[i, 1]-0.25))

red_point = embedding[[engine == 'Llama' for engine in metrics_df.Agent]]
green_point = embedding[[engine == 'Centaur' for engine in metrics_df.Agent]]

ax.text(-0.2, 1.09, 'a', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)

if red_point.size > 0 and green_point.size > 0:
    plt.arrow(
        red_point[0, 0], red_point[0, 1],
        green_point[0, 0] - red_point[0, 0], green_point[0, 1] - red_point[0, 1],
        head_width=0.4, head_length=0.4, overhang=0, fc='k', length_includes_head=True
    )

#plot feher da silva
twostep_centaur = pd.read_csv('../results/feher2023rethinking/tst_centaur_alignment.csv').values.mean(1)
twostep_llama = pd.read_csv('../results/feher2023rethinking/tst_llama_alignment.csv').values.mean(1)
baseline_model = 0.023144136642747695
print(twostep_llama)
print(twostep_centaur)

ax = fig.add_subplot(gs[:, 1])
ax.plot([0, 10, 20, 30, 40], twostep_centaur, color='#69005f', alpha=0.8, linewidth=1)
ax.plot([0, 10, 20, 30, 40], twostep_llama, color='#ff506e', alpha=0.8, linewidth=1)
ax.axhline(y=baseline_model, color='grey', linestyle='--', linewidth=1.0)
ax.text(41, baseline_model - 0.018, 'Cognitive model', fontsize=6, color='grey', horizontalalignment='right')
ax.text(-0.2, 1.09, 'b', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)
ax.set_ylabel(r'R$^2$')
ax.set_xlabel('Layer')
ax.set_xlim(1, 41)
ax.set_ylim(-0.02, 0.24)
ax.legend(['Centaur', 'Llama'], frameon=False, ncols=2, borderaxespad=0, handlelength=1, columnspacing=0.7, handletextpad=0.5, bbox_to_anchor=(0.51, 1.125), loc='upper center')

# plot tuckute
reading_llama = torch.load('../results/tuckute2024driving/llama.pth')
reading_centaur = torch.load('../results/tuckute2024driving/centaur2000.pth')

ax = fig.add_subplot(gs[:, 2])
ax.plot(torch.arange(1, reading_centaur.shape[0] + 1), reading_centaur, color='#69005f', alpha=0.8, linewidth=1)
ax.plot(torch.arange(1, reading_llama.shape[0] + 1), reading_llama, color='#ff506e', alpha=0.8, linewidth=1)
ax.axhline(y=0.38, color='grey', linestyle='--', linewidth=1.0)
ax.axhline(y=0.56, color='black', linestyle='--', linewidth=1.0)
ax.text(41, 0.357, 'Tuckute et al. (2024)', fontsize=6, color='grey', horizontalalignment='right')
ax.text(41, 0.57, 'Noise ceiling', fontsize=6, color='black', horizontalalignment='right')
ax.text(-0.2, 1.09, 'c', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)
ax.set_ylabel(r'R$^2$')
ax.set_xlabel('Layer',)
ax.set_xlim(1, 41)
ax.set_ylim(0.3, 0.63)
ax.legend(['Centaur', 'Llama'], frameon=False, ncols=2, borderaxespad=0, handlelength=1, columnspacing=0.7, handletextpad=0.5, bbox_to_anchor=(0.51, 1.125), loc='upper center')

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig4.pdf', bbox_inches='tight')

plt.show()

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
import math
from scipy import stats

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

ax = fig.add_subplot(gs[0, 0])
ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=25, alpha=0.8)
ax.set_xlabel('Embedding dimension 1')
ax.set_ylabel('Embedding dimension 2')
ax.set_xlim(-4, 6)
ax.set_ylim(-4, 6)

for i in range(embedding.shape[0]):
    if agent_names[i] == 'GPT-3.5':
        ax.annotate(agent_names[i], (-0.6 + embedding[i, 0], embedding[i, 1]+0.5), size=6)
    else:
        ax.annotate(agent_names[i], (0.45 + embedding[i, 0], embedding[i, 1]-0.25),  size=6)

red_point = embedding[[engine == 'Llama' for engine in metrics_df.Agent]]
green_point = embedding[[engine == 'Centaur' for engine in metrics_df.Agent]]

ax.text(-0.22, 1.09, 'a', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)

if red_point.size > 0 and green_point.size > 0:
    plt.arrow(
        red_point[0, 0], red_point[0, 1],
        green_point[0, 0] - red_point[0, 0], green_point[0, 1] - red_point[0, 1],
        head_width=0.4, head_length=0.4, overhang=0, fc='k', length_includes_head=True
    )

#plot feher da silva
df_centaur_tst = torch.load('../results/feher2023rethinking/schaefer_tst_centaur_alignment.pth')
df_llama_tst = torch.load('../results/feher2023rethinking/schaefer_tst_llama_alignment.pth')
df_random_tst = torch.load('../results/feher2023rethinking/schaefer_tst_random_alignment.pth')

df_centaur_tst = np.array([list(layer.values()) for layer in df_centaur_tst])
df_llama_tst = np.array([list(layer.values()) for layer in df_llama_tst])
df_random_tst = np.array([list(layer.values()) for layer in df_random_tst])

twostep_centaur = df_centaur_tst.mean((1, 2))
twostep_llama = df_llama_tst.mean((1, 2))
twostep_random = df_random_tst.mean((1, 2))
twostep_centaur_se = df_centaur_tst.std((1, 2)) / math.sqrt(df_centaur_tst.shape[1] * df_centaur_tst.shape[2])
twostep_llama_se = df_llama_tst.std((1, 2)) / math.sqrt(df_centaur_tst.shape[1] * df_centaur_tst.shape[2])
twostep_random_se = df_random_tst.std((1, 2)) / math.sqrt(df_random_tst.shape[1] * df_random_tst.shape[2])

print('Two-step task:')
for i in range(5):
    print(stats.ttest_ind(df_centaur_tst[i].flatten(), df_llama_tst[i].flatten(), alternative='greater'))

baseline_model = 0.20065425519568694
print(twostep_llama)
print(twostep_centaur)

ax = fig.add_subplot(gs[0, 1])
ax.errorbar([0, 10, 20, 30, 40], twostep_centaur, yerr=twostep_centaur_se, color='#69005f', alpha=0.8, linewidth=1)
ax.errorbar([0, 10, 20, 30, 40], twostep_llama, yerr=twostep_llama_se, color='#ff506e', alpha=0.8, linewidth=1)
ax.errorbar([0, 10, 20, 30, 40], twostep_random, yerr=twostep_random_se, color='grey', alpha=0.8, linewidth=1)
ax.legend(['Centaur', 'Llama', 'Control'], frameon=False, ncols=3, borderaxespad=0, handlelength=1, columnspacing=0.7, handletextpad=0.5, bbox_to_anchor=(0.51, 1.125), loc='upper center')
ax.axhline(y=baseline_model, color='#cbc9e2', linestyle='--', linewidth=1.0)
ax.text(41, baseline_model - 0.0185, 'Cognitive model', fontsize=6, color='#aeabcc', horizontalalignment='right')
ax.text(-0.2, 1.09, 'b', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)
ax.set_ylabel('Pearson correlation')
ax.set_xlabel('Layer')
ax.set_xlim(1, 41)
ax.set_ylim(0.16, 0.44)


# plot tuckute
reading_llama = torch.load('../results/tuckute2024driving/llama.pth')
reading_centaur = torch.load('../results/tuckute2024driving/centaur2000.pth')
reading_random = torch.load('../results/tuckute2024driving/random.pth')
reading_llama_sem = torch.load('../results/tuckute2024driving/llama_sem.pth')
reading_centaur_sem = torch.load('../results/tuckute2024driving/centaur2000_sem.pth')
reading_random_sem = torch.load('../results/tuckute2024driving/random_sem.pth')

best_centaur = reading_centaur.argmax()
best_llama = reading_llama.argmax()

ax = fig.add_subplot(gs[0, 2])
ax.errorbar(torch.arange(1, reading_centaur.shape[0] + 1), reading_centaur, yerr=reading_centaur_sem, color='#69005f', alpha=0.8, linewidth=1)
ax.errorbar(torch.arange(1, reading_llama.shape[0] + 1), reading_llama, yerr=reading_llama_sem, color='#ff506e', alpha=0.8, linewidth=1)
ax.errorbar(torch.arange(1, reading_random.shape[0] + 1), reading_random, yerr=reading_random_sem, color='grey', alpha=0.8, linewidth=1)
ax.legend(['Centaur', 'Llama', 'Control'], frameon=False, ncols=3, borderaxespad=0, handlelength=1, columnspacing=0.7, handletextpad=0.5, bbox_to_anchor=(0.51, 1.125), loc='upper center')
ax.axhline(y=0.38, color='#cbc9e2', linestyle='--', linewidth=1.0)
ax.axhline(y=0.56, color='black', linestyle='--', linewidth=1.0)
ax.text(41, 0.34, 'Tuckute et al. (2024)', fontsize=6, color='#aeabcc', horizontalalignment='right')
ax.text(41, 0.575, 'Noise ceiling', fontsize=6, color='black', horizontalalignment='right')
ax.text(-0.2, 1.09, 'c', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)
ax.set_ylabel('Pearson correlation')
ax.set_xlabel('Layer',)
ax.set_xlim(1, 41)
ax.set_ylim(0.08, 0.64)


sns.despine()
plt.tight_layout()
plt.savefig('figures/fig4_new.pdf', bbox_inches='tight')

plt.show()

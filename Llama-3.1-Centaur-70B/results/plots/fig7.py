import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D

gs = gridspec.GridSpec(1, 2, width_ratios=[0.5, 0.5])

agents = [
    'Centaur',
    'Llama',
    'Human',
]

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 3))

color_1 = '#69005f'
color_2 = '#ff506e'
custom_lines_r2 = [Line2D([0], [0], color=color_1, alpha=0.8, marker="o", linestyle='None', markersize=5), Line2D([0], [0], color=color_2, alpha=0.8, marker="o", linestyle='None', markersize=5)]

ax = fig.add_subplot(gs[:, 0])
df = pd.read_csv('../results/CogBench/performance.csv')
df = df[df['Agent'].isin(agents)]
ci = [col for col in df.columns if '_ci' in col]
not_ci = [col for col in df.columns if not '_ci' in col]
df_T = df[not_ci].drop(['Unnamed: 0'], axis=1).set_index('Agent').T
df.columns = df.columns.str.rstrip('_x')
df_T_ci = df[ci]
df_T_ci.columns = df_T_ci.columns.str.rstrip('_ci')
df_T_ci = df_T_ci.set_index('Agent').T
df_T = df_T.drop(columns=['Human'])
df_T.index = df_T.index.str.replace('Probabilistic Reasoning', 'Probabilistic reasoning')
df_T.index = df_T.index.str.replace('Horizon Task', 'Horizon task')
df_T.index = df_T.index.str.replace('Restless Bandit', 'Restless bandit')
df_T.index = df_T.index.str.replace('Instrumental Learning', 'Instrumental learning')
df_T.index = df_T.index.str.replace('BART', 'Balloon analog risk task')
df_T.index = df_T.index.str.replace('Two Step Task', 'Two-step task')
df_T_ci.index = df_T.index.str.replace('Probabilistic Reasoning', 'Probabilistic reasoning')
df_T_ci.index = df_T.index.str.replace('Horizon Task', 'Horizon task')
df_T_ci.index = df_T.index.str.replace('Restless Bandit', 'Restless bandit')
df_T_ci.index = df_T.index.str.replace('Instrumental Learning', 'Instrumental learning')
df_T_ci.index = df_T.index.str.replace('BART', 'Balloon analog risk task')
df_T_ci.index = df_T.index.str.replace('Two Step Task', 'Two-step task')
df_T = df_T[['Centaur', 'Llama']]
df_T_ci = df_T_ci[['Centaur', 'Llama']]
ax.text(-0.06, 1.2, 'a', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)
ax.text(5.5, 1.1, 'Humans', fontsize=6, color='grey', horizontalalignment='right')
ax.text(5.5, -0.3, 'Random', fontsize=6, color='grey', horizontalalignment='right')
df_T.plot(kind='bar', yerr=df_T_ci, ax=ax, legend=False, color=['#69005f', '#ff506e'], alpha=0.8)
ax.legend(custom_lines_r2, ['Centaur', 'Llama'], frameon=False, ncols=3, bbox_to_anchor=(0.5, 1.3), loc='upper center')
ax.set_ylim(-0.6, 2.2)
ax.set_ylabel('Performance')
ax.hlines(y=1, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)
ax.hlines(y=0, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)

ax = fig.add_subplot(gs[:, 1])
df = pd.read_csv('../results/CogBench/behaviour.csv')
df = df[df['Agent'].isin(agents)]
ci = [col for col in df.columns if '_ci' in col]
not_ci = [col for col in df.columns if not '_ci' in col]
df_T = df[not_ci].drop(['Unnamed: 0'], axis=1).set_index('Agent').T
df.columns = df.columns.str.rstrip('_x')
df_T_ci = df[ci]
df_T_ci.columns = df_T_ci.columns.str.rstrip('_ci')
df_T_ci = df_T_ci.set_index('Agent').T
df_T = df_T.drop(columns=['Human'])
df_T = df_T[['Centaur', 'Llama']]
df_T_ci = df_T_ci[['Centaur', 'Llama']]
ax.text(-0.06, 1.2, 'b', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')  # Add label (b)
ax.text(9.5, 1.1, 'Humans', fontsize=6, color='grey', horizontalalignment='right')
ax.text(9.5, -0.3, 'Random', fontsize=6, color='grey', horizontalalignment='right')
df_T.plot(kind='bar', yerr=df_T_ci, ax=ax,legend=False, color=['#69005f', '#ff506e'], alpha=0.8)
ax.set_ylim(-0.6, 2.1)
ax.legend(custom_lines_r2, ['Centaur', 'Llama'], frameon=False, ncols=3, bbox_to_anchor=(0.5, 1.3), loc='upper center')
ax.set_ylabel('Parameter value')
ax.hlines(y=1, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)
ax.hlines(y=0, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)

plt.tight_layout()
sns.despine()
plt.savefig('figures/fig7.pdf', bbox_inches='tight')
plt.show()

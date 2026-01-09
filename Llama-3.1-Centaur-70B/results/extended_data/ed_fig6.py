import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

gs = gridspec.GridSpec(2, 2, width_ratios=[0.5, 0.5])

def get_performance(file_name):
    with open(file_name) as f:
        d = json.load(f)
        k = d.keys()
        v = [d[key]['exact_match,strict-match'] if (key == 'metabench_gsm8k') else d[key]['acc,none'] for key in k]
        verr = [d[key]['exact_match_stderr,strict-match'] if (key == 'metabench_gsm8k') else d[key]['acc_stderr,none'] for key in k]
    return k, v, verr

#### METABENCH ####
k, v_llama, verr_llama = get_performance('../results/metabench/base-llama-3_1-70B-results.json')
_, v_centaur, verr_centaur = get_performance('../results/metabench/centaur-2000-results.json')

df = pd.DataFrame(
    {'Task': k,
     'Llama': v_llama,
     'Centaur': v_centaur
    }).set_index('Task')

df_ci = pd.DataFrame(
    {'Task': k,
     'Llama': verr_llama,
     'Centaur': verr_centaur
    }).set_index('Task')

df.index = df.index.str.replace('metabench_arc', 'ARC')
df.index = df.index.str.replace('metabench_gsm8k', 'GSM8K')
df.index = df.index.str.replace('metabench_hellaswag', 'HellaSwag')
df.index = df.index.str.replace('metabench_mmlu', 'MMLU')
df.index = df.index.str.replace('metabench_truthfulqa', 'TruthfulQA')
df.index = df.index.str.replace('metabench_winogrande', 'Winogrande')
df.index = df.index.str.replace('metabench', 'Mean')
df_ci.index = df_ci.index.str.replace('metabench_arc', 'ARC')
df_ci.index = df_ci.index.str.replace('metabench_gsm8k', 'GSM8K')
df_ci.index = df_ci.index.str.replace('metabench_hellaswag', 'HellaSwag')
df_ci.index = df_ci.index.str.replace('metabench_mmlu', 'MMLU')
df_ci.index = df_ci.index.str.replace('metabench_truthfulqa', 'TruthfulQA')
df_ci.index = df_ci.index.str.replace('metabench_winogrande', 'Winogrande')
df_ci.index = df_ci.index.str.replace('metabench', 'Mean')
df = df[['Centaur', 'Llama']]
df_ci = df_ci[['Centaur', 'Llama']]

print(df)
print(df_ci)
df_full = pd.merge(df, df_ci, how='inner', on=[df_ci.index])
print(df_full)
for _, row in df_full.iterrows():
    print()
    #df_ci_row = df_ci[df_ci[] == row.name]
    z_stat = (row['Centaur_x'] - row['Llama_x']) / ((row['Centaur_y'] ** 2 + row['Llama_y'] ** 2) ** 0.5)
    # Calculate p-value
    p_value = np.round(stats.norm.sf(abs(z_stat)) * 2, 3)  # two-tailed test

    print(f'{row["key_0"]}: $z = {z_stat}$, $p = {p_value}$')


plt.style.use(['nature'])
fig = plt.figure(figsize=(7.20472, 7))

ax1 = fig.add_subplot(gs[0, :])

df.plot(ax=ax1, kind='bar', yerr=df_ci, legend=False, color=['#69005f', '#ff506e'], alpha=0.8)
color_1 = '#69005f'
color_2 = '#ff506e'
custom_lines_r2 = [
    Line2D([0], [0], color=color_1, alpha=0.8, linewidth=5, markersize=2),
    Line2D([0], [0], color=color_2, alpha=0.8, linewidth=5, markersize=2)
]


plt.legend(custom_lines_r2, ['Centaur', 'Llama'], frameon=False, ncols=1, loc='upper right')
ax1.set_ylabel('Performance', labelpad=10)
ax1.set_xlabel('')
ax1.set_ylim(-0.0, 1.4)

#### COGBENCH ####
agents = [
    'Centaur',
    'Llama',
    'Human',
]

ax = fig.add_subplot(gs[1, 0])
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
df_full = pd.merge(df_T, df_T_ci, how='inner', on=[df_T_ci.index])
print(df_full)
for _, row in df_full.iterrows():
    #df_ci_row = df_ci[df_ci[] == row.name]
    z_stat = (row['Centaur_x'] - row['Llama_x']) / (((row['Centaur_y']/1.96) ** 2 + (row['Llama_y']/1.96) ** 2) ** 0.5)
    # Calculate p-value
    p_value = np.round(stats.norm.sf(abs(z_stat)), 3)  # one-tailed test
    z_stat = np.round(z_stat, 3)

    print(f'\item {row["key_0"]}: $z = {z_stat}$, $p = {p_value}$.')

ax.text(5.5, 1.05, 'Humans', fontsize=6, color='grey', horizontalalignment='right')
ax.text(5.5, -0.16, 'Random', fontsize=6, color='grey', horizontalalignment='right')
df_T.plot(kind='bar', yerr=df_T_ci/1.96, ax=ax, legend=False, color=['#69005f', '#ff506e'], alpha=0.8)
ax.legend(custom_lines_r2, ['Centaur', 'Llama'], frameon=False, ncols=1, loc='upper right')
ax.set_ylim(-0.6, 2.2)
ax.set_ylabel('Performance')
ax.hlines(y=1, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)
ax.hlines(y=0, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)

ax = fig.add_subplot(gs[1, 1])
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
df_full = pd.merge(df_T, df_T_ci, how='inner', on=[df_T_ci.index])
print(df_full)
for _, row in df_full.iterrows():
    #df_ci_row = df_ci[df_ci[] == row.name]
    z_stat = (row['Centaur_x'] - row['Llama_x']) / (((row['Centaur_y']/1.96) ** 2 + (row['Llama_y']/1.96) ** 2) ** 0.5)
    # Calculate p-value
    p_value = np.round(stats.norm.sf(abs(z_stat)), 3)  # one-tailed test
    z_stat = np.round(z_stat, 3)

    print(f'\item {row["key_0"]}: $z = {z_stat}$, $p = {p_value}$.')


ax.text(9.5, 1.05, 'Humans', fontsize=6, color='grey', horizontalalignment='right')
ax.text(9.5, -0.16, 'Random', fontsize=6, color='grey', horizontalalignment='right')
df_T.plot(kind='bar', yerr=df_T_ci/1.96, ax=ax,legend=False, color=['#69005f', '#ff506e'], alpha=0.8)
ax.set_ylim(-0.6, 2.1)
ax.legend(custom_lines_r2, ['Centaur', 'Llama'], frameon=False, ncols=1, loc='upper right')
ax.set_ylabel('Parameter value')
ax.hlines(y=1, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)
ax.hlines(y=0, xmin=-1, xmax=20, color='grey', linestyle='--', linewidth=1.0)

fig.text(0.012, 0.975, 'a', fontsize=8, weight='bold')
fig.text(0.012, 0.521, 'b', fontsize=8, weight='bold')
fig.text(0.505, 0.521, 'c', fontsize=8, weight='bold')

plt.tight_layout()
sns.despine()
plt.savefig('figures/fig6.jpg', bbox_inches='tight', dpi=300)
plt.show()

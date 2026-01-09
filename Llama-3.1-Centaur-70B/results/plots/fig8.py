import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scienceplots
import matplotlib.gridspec as gridspec
from functools import reduce
from brokenaxes import brokenaxes

selected_tasks  = ['peterson2021using', 'ruggeri2022globalizability']

df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['task'].isin(selected_tasks)][['task', 'random']]

df_llama_70b = pd.read_csv('../results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv')
df_llama_70b = df_llama_70b[df_llama_70b['task'].isin(selected_tasks)][['task', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit']]

df_centaur_70b = pd.read_csv('../results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df_centaur_70b = df_centaur_70b[df_centaur_70b['task'].isin(selected_tasks)][['task', 'marcelbinz/Llama-3.1-Centaur-70B-adapter']]

df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['task'].isin(selected_tasks)][['task', 'baseline']]

df_centaur_70b_no_history = pd.read_csv('../ceiling/results/marcelbinz-Llama-3.1-Centaur-70B-adapter.csv',  index_col=0)
df_centaur_70b_no_history = df_centaur_70b_no_history.rename(columns={'marcelbinz/Llama-3.1-Centaur-70B-adapter': 'marcelbinz/Llama-3.1-Centaur-70B-adapter-no-history'})
df_centaur_70b_no_history['task'] = df_centaur_70b_no_history['task'].str.replace('/prompts_zeroshot.jsonl','')

df_llama_70b_no_history = pd.read_csv('../ceiling/results/unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv', index_col=0)
df_llama_70b_no_history = df_llama_70b_no_history.rename(columns={'unsloth/Meta-Llama-3.1-70B-bnb-4bit': 'unsloth/Meta-Llama-3.1-70B-bnb-4bit-no-history'})
df_llama_70b_no_history['task'] = df_llama_70b_no_history['task'].str.replace('/prompts_zeroshot.jsonl','')

df_ceiling = pd.read_csv('../ceiling/results/ceiling.csv')
df_ceiling = df_ceiling.rename(columns={'nll': 'ceiling'})
df_ceiling = df_ceiling[df_ceiling['task'].isin(selected_tasks)][['task', 'ceiling']]

df = reduce(lambda left,right: pd.merge(left,right,on=['task'], how='outer'), [df_random, df_llama_70b, df_centaur_70b, df_centaur_70b_no_history, df_llama_70b_no_history, df_ceiling, df_baseline])
print(df)

model_names = ['random', 'marcelbinz/Llama-3.1-Centaur-70B-adapter', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit', 'baseline', 'ceiling', 'marcelbinz/Llama-3.1-Centaur-70B-adapter-no-history', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit-no-history']
offsets = [0.01, 0.01]

gs = gridspec.GridSpec(1, 2, width_ratios=[0.5, 0.5])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 1.9))
for task_index, task in enumerate(df['task']):
    print(task)
    df_task = df[df['task'] == task][model_names].values.flatten()
    df_task = df_task[[1, 2, 3, 4, 5, 6, 0]]
    df_task[df_task != df_task] = 0
    print(df_task)
    cutoff = (2.67, 2.72) if task_index == 0 else (2.0, 2.05)
    ax = brokenaxes(ylims=((.4, .74), cutoff), subplot_spec=gs[task_index])

    ax.bar(np.arange(6), df_task[:-1], color=['#69005f', '#ff506e', '#cbc9e2', 'white', '#69005f', '#ff506e'])
    ax.set_xticks(np.arange(6), ['Centaur', 'Llama', 'Cog.\nmodel', 'Noise\nceiling', 'Centaur\n(ind.)', 'Llama\n(ind.)',])
    ax.axhline(y=df_task[-1], color='grey', linestyle='--', linewidth=1.0)
    ax.axs[-1].text(5.65, df_task[-1] + offsets[task_index], 'Random guessing', fontsize=6, color='grey', horizontalalignment='right')
    ax.set_title('choices13k' if task_index == 0 else 'Intertemporal choice', fontsize=8)
    fig.axes[-1].text(-0.13, 1.12, 'a' if task_index == 0 else 'b', fontsize=8, fontweight='bold', va='top')
    if task_index == 0:
        ax.set_ylabel('Negative log-likelihood')
    
    print(ax.containers)
    ax.axs[-1].containers[0][0].set_alpha(0.8)
    ax.axs[-1].containers[0][1].set_alpha(0.8)
    ax.axs[-1].containers[0][2].set_alpha(1)
    ax.axs[-1].containers[0][3].set_alpha(0.5)
    ax.axs[-1].containers[0][4].set_alpha(0.8)
    ax.axs[-1].containers[0][5].set_alpha(0.8)
    ax.axs[-1].containers[0][3].set_edgecolor('black')
    ax.axs[-1].containers[0][4].set_hatch('///')
    ax.axs[-1].containers[0][5].set_hatch('///')

#sns.despine()
#plt.tight_layout()
plt.savefig('figures/fig8.pdf', bbox_inches='tight')
plt.show()

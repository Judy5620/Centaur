import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scienceplots
import matplotlib.gridspec as gridspec
from functools import reduce

plot_8b = False

df_llama_70b = pd.read_csv('../results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv')
df_llama_70b = df_llama_70b[df_llama_70b['unseen'] == 'experiments'][['task', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit']]

df_centaur_70b = pd.read_csv('../results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df_centaur_70b = df_centaur_70b[df_centaur_70b['unseen'] == 'experiments'][['task', 'marcelbinz/Llama-3.1-Centaur-70B-adapter']]

df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['unseen'] == 'experiments'][['task', 'baseline']]

df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['unseen'] == 'experiments'][['task', 'random']]

if plot_8b:
    df_llama_8b = pd.read_csv('../results/all_data_unsloth-Meta-Llama-3.1-8B-bnb-4bit.csv')
    df_llama_8b = df_llama_8b[df_llama_8b['unseen'] == 'experiments'][['task', 'unsloth/Meta-Llama-3.1-8B-bnb-4bit']]

    df_centaur_8b = pd.read_csv('../results/all_data_marcelbinz-Llama-3.1-Centaur-8B-adapter.csv')
    df_centaur_8b = df_centaur_8b[df_centaur_8b['unseen'] == 'experiments'][['task', 'marcelbinz/Llama-3.1-Centaur-8B-adapter']]

    df = reduce(lambda left,right: pd.merge(left,right,on=['task'], how='outer'), [df_llama_70b, df_centaur_70b, df_baseline, df_llama_8b, df_centaur_8b, df_random])
else:
    df = reduce(lambda left,right: pd.merge(left,right,on=['task'], how='outer'), [df_llama_70b, df_centaur_70b, df_baseline, df_random])

df2 =  pd.read_csv('../results/fig3_data.csv')
print(df2)
print(df)
#print(dfgdfgfd)
gs = gridspec.GridSpec(1, 3, width_ratios=[0.3333, 0.3333, 0.3333])
if plot_8b:
    model_names = ['random', 'marcelbinz/Llama-3.1-Centaur-70B-adapter', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit', 'baseline', 'marcelbinz/Llama-3.1-Centaur-8B-adapter', 'unsloth/Meta-Llama-3.1-8B-bnb-4bit']
else:
    model_names = ['random', 'marcelbinz/Llama-3.1-Centaur-70B-adapter', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit', 'baseline']

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 1.9))
for task_index, task in enumerate(df['task']):
    print(task)
    scale = 1 if task_index == 1 else 0.5
    df_task = df[df['task'] == task][model_names].values.flatten()
    ll_random = -df_task[0]
    df_task = 1 - (-df_task[1:]/ll_random)
    df_task[df_task != df_task] = 0
    print(df_task)
    ax = fig.add_subplot(gs[:, task_index])

    if plot_8b:
        ax.bar(np.arange(5), df_task, color=['#69005f', '#ff506e', '#cbc9e2', 'C0', 'C1'])
        ax.set_xticks(np.arange(5), ['Centaur\n(70B)', 'Llama\n(70B)', 'Cognitive\nmodel', 'Centaur\n(8B)', 'Llama\n(8B)'], size=5)
    else:
        ax.bar(np.arange(3), df_task, color=['#69005f', '#ff506e', '#cbc9e2'])
        ax.set_xticks(np.arange(3), ['Centaur', 'Llama', 'Cognitive\nmodel'])

    if task_index == 2:
        if plot_8b:
            ax.text(0.45, 0.15, 'N/A', transform=ax.transAxes, va='top')
        else:
            ax.text(0.775, 0.15, 'N/A', transform=ax.transAxes, va='top')
    if task_index == 0:
        ax.set_ylabel(r'Pseudo-R$^2$')
    ax.containers[0][0].set_alpha(0.8)
    ax.containers[0][1].set_alpha(0.8)
    ax.containers[0][2].set_alpha(1)
    if plot_8b:
        ax.containers[0][3].set_alpha(0.8)
        ax.containers[0][4].set_alpha(0.8)

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig3_8b=' + str(plot_8b) + '.pdf', bbox_inches='tight')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import scienceplots
import matplotlib.gridspec as gridspec
from functools import reduce
import torch
import math
from scipy import stats
import matplotlib as  mpl

def cohen_d(x,y):
    return (np.mean(x) -np. mean(y)) / math.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)

centaur_70b = torch.load('../generalization/results/generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama_70b = torch.load('../generalization/results/generalization_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')

df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['unseen'] == 'experiments'][['task', 'baseline']]

df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['unseen'] == 'experiments'][['task', 'random']]

means = {}
sems = {}
for key in centaur_70b.keys():
    print(key)
    print(centaur_70b[key].shape)
    baseline = df_baseline[df_baseline['task'] == key]
    random = df_random[df_random['task'] == key]
    means[key] = []
    sems[key] = []
    means[key].append(centaur_70b[key].mean())
    means[key].append(llama_70b[key].mean())
    sems[key].append(centaur_70b[key].std() / math.sqrt(len(centaur_70b[key])))
    sems[key].append(llama_70b[key].std() / math.sqrt(len(llama_70b[key])))


    print(stats.ttest_ind(centaur_70b[key], llama_70b[key], alternative='less'))
    print("Cohen's d:", cohen_d(centaur_70b[key], llama_70b[key]))

    if len(baseline) > 0:
        means[key].append(baseline.baseline.item())
        print(centaur_70b[key].shape)
        print(stats.ttest_1samp(centaur_70b[key], llama_70b[key].mean().item(), alternative='less'))
        print(stats.ttest_1samp(centaur_70b[key], baseline.baseline.item(), alternative='less'))
        print(stats.ttest_1samp(llama_70b[key], baseline.baseline.item(), alternative='less'))
    else:
        means[key].append(0)
    print(means)
    sems[key].append(0)
    means[key].append(random.random.item())
    sems[key].append(0)
    print()

plt.style.use(['nature'])
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "pdf.fonttype": 42,
    "text.usetex": False,
})
fig = plt.figure(figsize=(7.20472, 4.5))
gs = gridspec.GridSpec(2, 3, width_ratios=[0.3333, 0.3333, 0.3333], height_ratios=[1.25, 1])

ax = fig.add_subplot(gs[0, 0])
#image = plt.imread('tstcover.png')
#cax = ax.imshow(image)
ax.axis('off')
ax.set_title('Modified cover story', fontsize=7)

ax = fig.add_subplot(gs[0, 1])
#image = plt.imread('bandit.png')
#cax = ax.imshow(image)
ax.axis('off')
ax.set_title('Modified problem structure', fontsize=7)

ax = fig.add_subplot(gs[0, 2])
#image = plt.imread('logical2.png')
#cax = ax.imshow(image)
ax.axis('off')
ax.set_title('Entirely novel domain', fontsize=7)


#print(dfgdfgfd)

offsets = [0.007, 0.021, 0.020]
for task_index, task in enumerate(means.keys()):
    print(task)
    ax = fig.add_subplot(gs[1, task_index])
    ax.bar(np.arange(3), means[task][:-1], yerr=sems[task][:-1], color=['#69005f', '#ff506e', '#cbc9e2'])
    ax.set_xticks(np.arange(3), ['Centaur', 'Llama', 'Cognitive\nmodel'])
    ax.axhline(y=means[task][-1], color='grey', linestyle='--', linewidth=1.0)
    ax.text(2.5, means[task][-1] + offsets[task_index], 'Random guessing', fontsize=6, color='grey', horizontalalignment='right')

    if task_index == 2:
        ax.text(0.775, 0.125, 'N/A', transform=ax.transAxes, va='top')

    ax.set_ylabel('Negative log-likelihood')
    ax.containers[1][0].set_alpha(0.8)
    ax.containers[1][1].set_alpha(0.8)
    ax.containers[1][2].set_alpha(1)
    ax.set_ylim(0.9  * means[task][0], 1.1 * means[task][-1])

fig.text(0.005, 0.961, 'a', fontsize=8, weight='bold')
fig.text(0.34, 0.961, 'b', fontsize=8, weight='bold')
fig.text(0.67, 0.961, 'c', fontsize=8, weight='bold')

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig3_part1.pdf', bbox_inches='tight')
plt.show()

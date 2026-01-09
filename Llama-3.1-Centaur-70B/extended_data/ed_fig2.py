import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from glob import glob
import scienceplots
import matplotlib.gridspec as gridspec
import scipy.stats as st
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from functools import reduce
import torch
import math
from scipy import stats

def cohen_d(x,y):
    return (np.mean(x) -np. mean(y)) / math.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.20472, 5))

color_1 = '#69005f'
color_2 = '#ff506e'
color_3 = '#cbc9e2'
color_4 = 'C0'
color_5 = 'C1'

cmap1 = LinearSegmentedColormap.from_list("", ["white", color_1])
new_cmap1 = truncate_colormap(cmap1, 0.2, 1.0)
new_cmap0 = truncate_colormap(plt.get_cmap('Greys'), 0.2, 1.0)
gs = gridspec.GridSpec(3, 2, width_ratios=[0.6666, 0.3333])

centaur_70b = torch.load('../results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama_70b = torch.load('../results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-8B-adapter.pth')


df_exp = pd.read_csv('../experiments.csv', sep=';')
baselines_full = torch.load('../results/custom_metrics_full_log_likelihoods_baselines.pth')

papers = []
results_centaur = []
results_llama = []
ll_centaur = []
ll_llama = []
ll_baseline = []
baselines = []
for key in baselines_full.keys():
    baseline = baselines_full[key].mean().item()
    papers.append(df_exp[df_exp['path'] == key + '/']['task_name'].item())
    print(key)
    print(len(centaur_70b[key]))
    print(centaur_70b[key].mean())
    print(baseline)
    print()
    centaur_relative = -centaur_70b[key] + baseline
    llama_relative = -llama_70b[key] + baseline
    results_centaur.append(centaur_relative)
    results_llama.append(llama_relative)
    ll_centaur.append(centaur_70b[key])
    ll_llama.append(llama_70b[key])
    ll_baseline.append(baselines_full[key])
    baselines.append(baseline)

centaur_full = np.concatenate(ll_centaur)
llama_full = np.concatenate(ll_llama)
baseline_full = np.concatenate(ll_baseline)
print('Baselines: ', baseline_full.mean())
print('Centaur: ', centaur_full.mean())
print('Centaur var: ', centaur_full.var())
print('Llama: ', llama_full.mean())
print('Llama var: ', llama_full.var())
print('Centaur SEM: ', centaur_full.std() / math.sqrt(len(centaur_full)))
print('Llama SEM:', llama_full.std() / math.sqrt(len(llama_full)))
print(stats.ttest_ind(centaur_full, llama_full, alternative='less'))
results_centaur_copy = results_centaur
results_llama_copy = results_llama

print(baseline_full)
print(stats.ttest_ind(centaur_full, baseline_full, alternative='less'))
print(stats.ttest_1samp(centaur_full, baseline_full.mean(), alternative='less'))
print("Cohen's d:", cohen_d(centaur_full, llama_full))
print("Cohen's d:", cohen_d(centaur_full, baseline_full))

deduped_centaur = {}
deduped_llama = {}
for i, paper in enumerate(papers):
    if paper in deduped_centaur.keys():
        deduped_centaur[paper] =  np.concatenate((deduped_centaur[paper], results_centaur[i]))
        deduped_llama[paper] =  np.concatenate((deduped_llama[paper], results_llama[i]))
    else:
        deduped_centaur[paper] = results_centaur[i]
        deduped_llama[paper] = results_llama[i]

papers = []
results_centaur = []
results_centaur_se = []
results_llama = []
for paper in deduped_centaur.keys():
    papers.append(paper)
    print(deduped_centaur[paper].shape)
    results_centaur_se.append(deduped_centaur[paper].std() / math.sqrt(len(deduped_centaur[paper])))
    results_llama.append(deduped_llama[paper].mean())
    results_centaur.append(deduped_centaur[paper].mean())

ax1 = fig.add_subplot(gs[:, 0])

order = np.argsort(results_centaur)
papers = np.array(papers)[order]
results_centaur = np.array(results_centaur)[order]
results_llama = np.array(results_llama)[order]

print(len(np.unique(papers)))
print(len(results_centaur))

ax1.barh([len(results_centaur) + 1 ], [np.concatenate(results_centaur_copy).mean()], xerr=[np.concatenate(results_centaur_copy).std() / math.sqrt(len(np.concatenate(results_centaur_copy)))],  height=0.75, color=color_1, alpha=0.8)
ax1.barh(np.arange(len(results_centaur)), results_centaur, xerr=results_centaur_se, height=0.75, color=color_1, alpha=0.8)


custom_lines_r2 = [
    Line2D([0], [0], color=color_1, alpha=0.8, linewidth=5, markersize=5),
    Line2D([0], [0], color=color_1, alpha=0.5, linewidth=5, markersize=5),
    Line2D([0], [0], color=color_3, linestyle='dashed', markersize=5)]
ax1.barh([len(results_llama) + 1], [np.concatenate(results_llama_copy).mean()], height=0.75, color='white', alpha=0.3)
ax1.barh(np.arange(len(results_llama)), results_llama, height=0.75, color='white', alpha=0.3)


ax1.set_yticks(np.arange(len(results_centaur)).tolist() + [len(results_centaur) + 1], papers.tolist() + ['Overall'])
ax1.set_xlabel(r'$\Delta$ log-likelihood')
ax1.set_ylim(-0.5, len(results_centaur) + 2)
ax1.axvline(x=0, color='grey', linestyle='--', linewidth=1.0)

ax1.legend(custom_lines_r2, ['Centaur', 'Minitaur', 'Cognitive model'], frameon=False, ncols=1)

centaur_70b = torch.load('../generalization/results/generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama_70b = torch.load('../generalization/results/generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-8B-adapter.pth')

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

    if len(baseline) > 0:
        means[key].append(baseline.baseline.item())
    else:
        means[key].append(0)
    sems[key].append(0)
    means[key].append(random.random.item())
    sems[key].append(0)
    print()

offsets = [0.0125, 0.03, 0.03]
titles = ['Modified cover story', 'Modified problem structure', 'Entirely novel domain']
for task_index, task in enumerate(means.keys()):
    print(task)
    ax = fig.add_subplot(gs[task_index, 1])
    ax.bar(np.arange(3), means[task][:-1], yerr=sems[task][:-1], color=['#69005f', '#69005f', '#cbc9e2'])
    ax.set_xticks(np.arange(3), ['Centaur', 'Minitaur', 'Cognitive\nmodel'], size=6)
    ax.set_title(titles[task_index], size=7)
    ax.axhline(y=means[task][-1], color='grey', linestyle='--', linewidth=1.0)
    ax.text(2.5, means[task][-1] + offsets[task_index], 'Random guessing', fontsize=6, color='grey', horizontalalignment='right')

    if task_index == 2:
        ax.text(0.775, 0.15, 'N/A', transform=ax.transAxes, va='top')

    ax.set_ylabel('Negative log-likelihood')
    ax.containers[1][0].set_alpha(0.8)
    ax.containers[1][1].set_alpha(0.5)
    ax.containers[1][2].set_alpha(1)

    ax.set_ylim(0.9  * means[task][0], 1.1 * means[task][-1])

fig.text(0.012, 0.965, 'a', fontsize=8, weight='bold')
fig.text(0.72, 0.965, 'b', fontsize=8, weight='bold')
fig.text(0.72, 0.648, 'c', fontsize=8, weight='bold')
fig.text(0.72, 0.327, 'd', fontsize=8, weight='bold')

plt.tight_layout()
sns.despine()
plt.savefig('figures/fig2.jpg', bbox_inches='tight', dpi=300)
plt.show()

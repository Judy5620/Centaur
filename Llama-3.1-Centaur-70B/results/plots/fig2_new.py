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


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661, 5))

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

use_8b = False
if use_8b:
    llama_70b = torch.load('../results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-8B-adapter.pth')
else:
    llama_70b = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')

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
    #centaur_relative = ((-centaur_70b[key] + baseline) / baseline) * 100
    #llama_relative = ((-llama_70b[key] + baseline) / baseline) * 100
    centaur_relative = -centaur_70b[key] + baseline
    llama_relative = -llama_70b[key] + baseline
    results_centaur.append(centaur_relative)
    results_llama.append(llama_relative)
    ll_centaur.append(centaur_70b[key])
    ll_llama.append(llama_70b[key])
    ll_baseline.append(baselines_full[key])
    baselines.append(baseline)

#print('Different to Llama: ', (results_centaur - results_llama).mean())
centaur_full = np.concatenate(ll_centaur)
llama_full = np.concatenate(ll_llama)
baseline_full = np.concatenate(ll_baseline)
print('Baselines: ', baseline_full.mean())
print('Centaur: ', centaur_full.mean())
print('Llama: ', llama_full.mean())
print('Centaur SEM: ', centaur_full.std() / math.sqrt(len(centaur_full)))
print('Llama SEM:', llama_full.std() / math.sqrt(len(llama_full)))
print(stats.ttest_ind(centaur_full, llama_full, alternative='less'))
results_centaur_copy = results_centaur
results_llama_copy = results_llama


print(stats.ttest_1samp(centaur_full, baseline_full.mean(), alternative='less'))

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

if use_8b:
    custom_lines_r2 = [
        Line2D([0], [0], color=color_1, alpha=0.8, linewidth=5, markersize=5),
        Line2D([0], [0], color=color_1, alpha=0.5, linewidth=5, markersize=5),
        Line2D([0], [0], color=color_3, linestyle='dashed', markersize=5)]
    ax1.barh([len(results_llama) + 1], [np.concatenate(results_llama_copy).mean()], height=0.75, color='white', alpha=0.3)
    ax1.barh(np.arange(len(results_llama)), results_llama, height=0.75, color='white', alpha=0.3)
else:
    custom_lines_r2 = [
        Line2D([0], [0], color=color_1, alpha=0.8, linewidth=5, markersize=5),
        Line2D([0], [0], color=color_2, alpha=0.8, linewidth=5, markersize=5),
        Line2D([0], [0], color=color_3, linestyle='dashed', markersize=5)]
    ax1.barh([len(results_llama) + 1], [np.concatenate(results_llama_copy).mean()], height=0.75, color=color_2, alpha=0.8)
    ax1.barh(np.arange(len(results_llama)), results_llama, height=0.75, color=color_2, alpha=0.8)

ax1.set_yticks(np.arange(len(results_centaur)).tolist() + [len(results_centaur) + 1], papers.tolist() + ['Overall'])
ax1.set_xlabel(r'$\Delta$ log-likelihood')
#ax1.set_xlim(-45, 105)
ax1.set_ylim(-0.5, len(results_centaur) + 2)
ax1.axvline(x=0, color='grey', linestyle='--', linewidth=1.0)
if use_8b:
    ax1.legend(custom_lines_r2, ['Centaur', 'Minitaur', 'Cognitive model'], frameon=False, ncols=1)
else:
    ax1.legend(custom_lines_r2, ['Centaur', 'Llama', 'Cognitive model'], frameon=False, ncols=1)

if use_8b:
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
        ax.set_title(titles[task_index], size=6)
        ax.axhline(y=means[task][-1], color='grey', linestyle='--', linewidth=1.0)
        ax.text(2.5, means[task][-1] + offsets[task_index], 'Random guessing', fontsize=6, color='grey', horizontalalignment='right')

        if task_index == 2:
            ax.text(0.775, 0.15, 'N/A', transform=ax.transAxes, va='top')

        ax.set_ylabel('Negative log-likelihood')
        ax.containers[1][0].set_alpha(0.8)
        ax.containers[1][1].set_alpha(0.5)
        ax.containers[1][2].set_alpha(1)

        ax.set_ylim(0.9  * means[task][0], 1.1 * means[task][-1])

else:
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    # subplot 2
    centaur_rewards = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_centaur_twostep' + str(i) + '.csv')['reward'].values for i in range(1, 3)])
    human_rewards = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_human_twostep' + str(i) + '.csv')['reward'].values for i in range(1, 3)])
    rewards = np.concatenate((human_rewards, centaur_rewards))

    centaur_mb = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_centaur_twostep' + str(i) + '.csv')['param'].values for i in range(1, 3)])
    human_mb = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_human_twostep' + str(i) + '.csv')['param'].values for i in range(1, 3)])
    model_basednesses = np.concatenate((human_mb, centaur_mb))

    df = pd.DataFrame({'rewards': rewards, 'model_basednesses': model_basednesses, 'agent': ['Humans'] * 30 + ['Centaur 3.1'] * 30})
    df['rewards'] = df['rewards'] * 2
    humans = df.loc[df.agent == "Humans"]
    centaur = df.loc[df.agent == "Centaur 3.1"]

    custom_lines = [Line2D([0], [0], color=color_1, alpha=0.8, marker="o", linestyle='None', markersize=5), Line2D([0], [0], color='grey', marker="o", linestyle='None', markersize=5)]
    sns.kdeplot(x=humans.rewards, y=humans.model_basednesses, cmap=new_cmap0, shade=True, shade_lowest=False, ax=ax2, alpha=0.75)
    sns.kdeplot(x=centaur.rewards, y=centaur.model_basednesses, cmap=new_cmap1, shade=True, shade_lowest=False, ax=ax2, alpha=0.75)
    ax2.set_xlabel('Reward')
    ax2.set_xlim(0.1, 0.9)
    ax2.set_ylabel('Model-basedness')
    ax2.set_yticks([-0.5, 0, 0.5, 1.0, 1.5], [-0.5, 0, 0.5, 1.0, 1.5])
    ax2.text(-0.13, 1.12, 'c', transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top')
    ax1.text(-1.52, 35.02, 'a', fontsize=8, fontweight='bold', va='top')

    # subplot 3
    centaur_rewards = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_centaur_horizon' + str(i) + '.csv')['reward'].values for i in range(1, 5)])
    human_rewards = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_human_horizon' + str(i) + '.csv')['reward'].values for i in range(1, 5)])
    rewards = np.concatenate((human_rewards, centaur_rewards))

    centaur_ib = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_centaur_horizon' + str(i) + '.csv')['param'].values for i in range(1, 5)])
    human_ib = np.concatenate([pd.read_csv('../openloop/results/baselines_openloop_human_horizon' + str(i) + '.csv')['param'].values for i in range(1, 5)])
    information_bonus = np.concatenate((human_ib, centaur_ib))

    df = pd.DataFrame({'rewards': rewards, 'information_bonus': information_bonus, 'agent': ['Humans'] * 28 + ['Centaur 3.1'] * 28})
    humans = df.loc[df.agent == "Humans"]
    centaur = df.loc[df.agent == "Centaur 3.1"]

    sns.kdeplot(x=humans.rewards, y=humans.information_bonus, cmap=new_cmap0, shade=True, shade_lowest=False, ax=ax3, alpha=0.75)
    sns.kdeplot(x=centaur.rewards, y=centaur.information_bonus, cmap=new_cmap1, shade=True, shade_lowest=False, ax=ax3, alpha=0.75)
    ax3.set_xlim(42, 62)
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Information bonus')
    ax3.legend(custom_lines, ['Centaur', 'Humans'], columnspacing=0.7, frameon=False, ncols=2, bbox_to_anchor=(0.5, 1.22), loc='upper center')
    ax3.text(-0.13, 1.12, 'b', transform=ax3.transAxes, fontsize=8, fontweight='bold', va='top')

    # subplot 4
    df = pd.read_csv('../openloop/baar2021latent/gameDat.csv')
    df['correct'] = df['CorrAns'] == df['GivenAns']
    df = df.groupby('subID').filter(lambda x: ~((x.GivenAns == 'coop').all()))
    df = df.groupby('subID').filter(lambda x: ~((x.GivenAns == 'def').all()))

    df_nat = df[df['Variant'] == 'nat']
    df_inv = df[df['Variant'] == 'inv']

    humans_nat = df_nat.groupby('subID')['correct'].mean().values
    humans_inv = df_inv.groupby('subID')['correct'].mean().values

    df = pd.read_csv('../openloop/baar2021latent/simulation_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
    df['correct'] = df['CorrAns'] == df['GivenAns']
    df = df.groupby('subID').filter(lambda x: ~((x.GivenAns == 'coop').all()))
    df = df.groupby('subID').filter(lambda x: ~((x.GivenAns == 'def').all()))

    df_nat = df[df['Variant'] == 'nat']
    df_inv = df[df['Variant'] == 'inv']

    centaur_nat = df_nat.groupby('subID')['correct'].mean().values
    centaur_inv = df_inv.groupby('subID')['correct'].mean().values
    '''
    df_centaur = pd.read_csv('../openloop/jansen2021dunningkruger/simulation.csv')
    df_human = pd.read_csv("../openloop/jansen2021dunningkruger/exp1.csv")
    df_human =  df_human.head(len(df_centaur))

    centaur_predicted_scores = df_centaur[df_centaur['question'] == 'absAssess0'].choice.astype('float').values
    human_predicted_scores =  df_human[df_human['question'] == 'absAssess0'].choice.astype('float').values
    centaur_scores = df_centaur[df_centaur['question'] == 'score'].choice.astype('float').values
    human_scores =  df_human[df_human['question'] == 'score'].choice.astype('float').values
    scores = np.concatenate((human_scores, centaur_scores))
    predicted_scores = np.concatenate((human_predicted_scores, centaur_predicted_scores))
    df = pd.DataFrame({'predicted_scores': predicted_scores, 'scores': scores, 'agent': ['Humans'] * 1000 + ['Centaur 3.1'] * 1000})
    humans = df.loc[df.agent == "Humans"]
    centaur = df.loc[df.agent == "Centaur 3.1"]'''

    sns.kdeplot(x=humans_nat, y=humans_inv, cmap=new_cmap0, shade=True, shade_lowest=False, ax=ax4, alpha=0.75)
    sns.kdeplot(x=centaur_nat, y=centaur_inv, cmap=new_cmap1, shade=True, shade_lowest=False, ax=ax4, alpha=0.75)
    ax4.set_xlabel('Accuracy (human strategies)')
    ax4.set_ylabel('Accuracy (artificial strategies)')
    ax4.set_xlim(0, 1)
    ax4.set_xlim(0, 1)
    ax4.set_xticks([0, 0.5, 1.0], [0, 0.5, 1.0])
    ax4.set_yticks([0, 0.5, 1.0], [0, 0.5, 1.0])
    ax4.text(-0.13, 1.12, 'd', transform=ax4.transAxes, fontsize=8, fontweight='bold', va='top')

plt.tight_layout()
sns.despine()
plt.savefig('figures/fig2_new_8b=' + str(use_8b) + '.pdf', bbox_inches='tight')
plt.show()

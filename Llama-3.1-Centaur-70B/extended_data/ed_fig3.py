import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

centaur_70b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
centaur_8b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-8B-adapter.pth')
llama_70b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')
llama_8b = torch.load('../generalization/results/additional_generalization_full_log_likelihoods_unsloth-Meta-Llama-3.1-8B-bnb-4bit.pth')

nll_random = {
    'additional_experiments/awad2018moral.jsonl':  -math.log(1/2),
    'additional_experiments/demircan2024evaluatingcategory.jsonl':  -math.log(1/2),
    'additional_experiments/demircan2024evaluatingreward.jsonl':  -math.log(1/2),
    'additional_experiments/akata2023repeatedgames.jsonl':  -math.log(1/2),
    'additional_experiments/singh2022representing.jsonl':  -math.log(1/7),
    'additional_experiments/xu2021novelty.jsonl':  -math.log(1/3),
}

task_names = {
    'additional_experiments/awad2018moral.jsonl': 'Moral decision-making',
    'additional_experiments/demircan2024evaluatingcategory.jsonl':  'Naturalistic category learning',
    'additional_experiments/demircan2024evaluatingreward.jsonl':  'Naturalistic reward learning',
    'additional_experiments/akata2023repeatedgames.jsonl':  'Economic games',
    'additional_experiments/singh2022representing.jsonl':  'Behavioral propensities',
    'additional_experiments/xu2021novelty.jsonl': 'Deep sequential decision task',
}

gs = gridspec.GridSpec(2, 3, width_ratios=[0.3333, 0.3333, 0.3333])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.20472, 3.8))

offsets = [0.01, 0.01, 0.005, 0.01, 0.02, 0.01]
for i,  key in enumerate(centaur_70b.keys()):
    print(key)
    centaur_70b_r2 = centaur_70b[key].mean().item()
    centaur_8b_r2 = centaur_8b[key].mean().item()
    llama_70b_r2 = llama_70b[key].mean().item()
    llama_8b_r2 = llama_8b[key].mean().item()
    centaur_70b_r2_se = centaur_70b[key].std().item() / math.sqrt(len(centaur_70b[key]))
    centaur_8b_r2_se = centaur_8b[key].std().item() / math.sqrt(len(centaur_8b[key]))
    llama_70b_r2_se = llama_70b[key].std().item()  / math.sqrt(len(llama_70b[key]))
    llama_8b_r2_se = llama_8b[key].std().item()  / math.sqrt(len(llama_8b[key]))
    res = stats.ttest_ind(centaur_70b[key], llama_70b[key], alternative='less')

    print('t(' + str(int(res.df)) + ') = ' + str(np.round(res.statistic, 2)) + ', p = ' + str(np.round(res.pvalue, 6)))

    ax = fig.add_subplot(gs[0 if i < 3 else 1, i % 3])
    values = np.array([centaur_70b_r2, centaur_8b_r2, llama_70b_r2, llama_8b_r2])
    ax.bar(np.arange(4), values, yerr=[centaur_70b_r2_se, centaur_8b_r2_se, llama_70b_r2_se, llama_8b_r2_se], color=['#69005f', '#69005f', '#ff506e', '#ff506e'])# 'C0', 'C1'
    ax.set_xticks(np.arange(4), ['Centaur', 'Minitaur', 'Llama\n(70B)', 'Llama\n(8B)'])
    ax.axhline(y=nll_random[key], color='grey', linestyle='--', linewidth=1.0)
    ax.text(3.5, nll_random[key] + offsets[i], 'Random guessing', fontsize=6, color='grey', horizontalalignment='right')



    if i == 2:
        ax.set_ylim(0.58, 0.82)
        ax.set_yticks([0.6, 0.7, 0.8])
    else:
        ax.set_ylim(0.9  * min(nll_random[key], min(values)), 1.1 * max(max(values), nll_random[key]))

    ax.set_ylabel('Negative log-likelihood')
    ax.containers[1][0].set_alpha(0.8)
    ax.containers[1][1].set_alpha(0.5)
    ax.containers[1][2].set_alpha(0.8)
    ax.containers[1][3].set_alpha(0.5)
    ax.set_title(task_names[key], fontsize=7)

fig.text(0.012, 0.955, 'a', fontsize=8, weight='bold')
fig.text(0.012, 0.465, 'b', fontsize=8, weight='bold')
fig.text(0.344, 0.955, 'c', fontsize=8, weight='bold')
fig.text(0.344, 0.465, 'd', fontsize=8, weight='bold')
fig.text(0.67, 0.955, 'e', fontsize=8, weight='bold')
fig.text(0.67, 0.465, 'f', fontsize=8, weight='bold')

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig3.jpg', bbox_inches='tight', dpi=300)
plt.show()

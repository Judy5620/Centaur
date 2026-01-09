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
from datasets import load_dataset
import matplotlib as  mpl

test_data = load_dataset("marcelbinz/Psych-101-test")['test']
test_participants = test_data.filter(lambda example: example['experiment'] == "hilbig2014generalized/exp1.csv")['participant']
test_participants = [int(a) for a in test_participants]

nll_centaur = torch.stack(torch.load('data/log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth', weights_only=True))
AIC_centaur_test = 2 * nll_centaur[test_participants].sum().item()
print('Centaur AIC (test):', AIC_centaur_test)


nll_cog = torch.load('data/cognitive_nlls.pth', weights_only=True)
print(nll_cog.shape)
AIC_cog = 2 * torch.Tensor([1, 1, 1, 1, 2]).unsqueeze(-1) + 2 * nll_cog[:5].sum(-1)
print(AIC_cog[:, test_participants])
AIC_cog_test = AIC_cog[:, test_participants].sum(-1)
print('Cognitive model AIC (test):', AIC_cog_test)

#AIC_cog_test = torch.cat([AIC_cog_test[:3], torch.zeros(3), AIC_cog_test[:4], torch.zeros(3), AIC_cog_test])
labels = [
    'Weighted-additive strategy',
    'Equal  weighting',
    'Take-the-best heuristic',
    'DeepSeek-R1 discovered',
    'Scientific regret minimization'
]
print(AIC_cog_test)


plt.style.use(['nature'])
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "pdf.fonttype": 42,
    "text.usetex": False,
})
gs = gridspec.GridSpec(1, 5, width_ratios=[0.175, 0.1, 0.25, 0.1, 0.325])
fig = plt.figure(figsize=(7.20472, 2.8))

ax = fig.add_subplot(gs[0, 0])
ax.bar(np.arange(AIC_cog_test[:3].shape[0]), AIC_cog_test[:3], color='#cbc9e2', width=0.75)
ax.set_xticks(np.arange(AIC_cog_test[:3].shape[0]), labels[:3], rotation=90)
ax.axhline(y=AIC_centaur_test, color='#69005f', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_ylabel('AIC')
ax.set_ylim(0, 420)


ax = fig.add_subplot(gs[0, 2])
ax.bar(np.arange(AIC_cog_test[:4].shape[0]), AIC_cog_test[:4], color='#cbc9e2', width=0.75)
ax.set_xticks(np.arange(AIC_cog_test[:4].shape[0]), labels[:4], rotation=90)
ax.axhline(y=AIC_centaur_test, color='#69005f', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_ylabel('AIC')
ax.set_ylim(0, 420)

ax = fig.add_subplot(gs[0, 4])
ax.bar(np.arange(AIC_cog_test[:5].shape[0]), AIC_cog_test[:5], color='#cbc9e2', width=0.75)
ax.set_xticks(np.arange(AIC_cog_test[:5].shape[0]), labels[:5], rotation=90)
ax.axhline(y=AIC_centaur_test, color='#69005f', linestyle='--', linewidth=1.0, alpha=0.8)
ax.set_ylabel('AIC')
ax.set_ylim(0, 420)

fig.text(0.012, 0.955, 'a', fontsize=8, weight='bold')
fig.text(0.338, 0.955, 'b', fontsize=8, weight='bold')
fig.text(0.714, 0.955, 'c', fontsize=8, weight='bold')
fig.text(0.988, 0.661, 'Centaur', fontsize=6, color='#69005f', horizontalalignment='right', alpha=0.8)

'''
plt.ylabel('AIC')
plt.axhline(y=AIC_centaur_test, xmin=0, xmax=0.168, color='#69005f', linestyle='--', linewidth=1.0, alpha=0.8)
plt.axhline(y=AIC_centaur_test, xmin=0.332, xmax=0.555, color='#69005f', linestyle='--', linewidth=1.0, alpha=0.8)
plt.axhline(y=AIC_centaur_test, xmin=0.72, xmax=1, color='#69005f', linestyle='--', linewidth=1.0, alpha=0.8)

plt.xticks([0, 1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 17], labels, rotation=90)
plt.ylim(0)
plt.xlim(-0.5, 17.5)'''


sns.despine()
plt.tight_layout()
plt.savefig('figures/fig5_part1.pdf', bbox_inches='tight')
plt.show()

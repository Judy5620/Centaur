import torch
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns


Bs = torch.load('../contamination/results/Bs.pth')

log_Bs = torch.log(Bs)

plt.figure(figsize=(7.08661, 1.9))
plt.style.use(['nature'])
plt.scatter(torch.arange(len(log_Bs)), log_Bs, color='#69005f')
plt.axhline(y=1, color='grey', linestyle='--', linewidth=1.0)

plt.text(len(log_Bs), 1.1, 'potentially contaminated', fontsize=6, color='red', horizontalalignment='right')
plt.text(len(log_Bs), 0.8, 'not contaminated', fontsize=6, color='green', horizontalalignment='right')
plt.ylabel(r'$\log B$')
plt.xlabel('Experiment')
plt.ylim(-1.6, 1.1)
plt.xlim(-0.5, len(log_Bs)+0.1)
sns.despine()
plt.tight_layout()
plt.savefig('figures/fig9.pdf', bbox_inches='tight')
plt.show()

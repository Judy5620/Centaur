import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.gridspec as gridspec
import numpy as np

hermes = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Hermes-3-Llama-3.1-70B-bnb-4bit.pth')
nemotron = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Llama-3.1-Nemotron-70B-Instruct-bnb-4bit.pth')
reflection = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Reflection-Llama-3.1-70B-bnb-4bit.pth')
instruct = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-llama-3-70b-Instruct-bnb-4bit.pth')
centaur = torch.load('../results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')

results_centaur = []
results_llama = []
results_instruct = []
results_nemotron = []
results_hermes = []
results_reflection = []
for key in centaur.keys():
        print(key)
        results_centaur.append(centaur[key])
        results_llama.append(llama[key])
        results_instruct.append(instruct[key])
        results_nemotron.append(nemotron[key])
        results_hermes.append(hermes[key])
        results_reflection.append(reflection[key])

results_centaur = np.concatenate(results_centaur)
results_llama = np.concatenate(results_llama)
results_instruct = np.concatenate(results_instruct)
results_nemotron = np.concatenate(results_nemotron)
results_hermes = np.concatenate(results_hermes)
results_reflection = np.concatenate(results_reflection)

gs = gridspec.GridSpec(1, 1, width_ratios=[1])
plt.style.use(['nature'])
fig = plt.figure(figsize=(7.08661/2, 1.9))


ax = fig.add_subplot(gs[0, :])
means = [
    np.array(results_centaur).mean(),
    np.array(results_llama).mean(),
    #np.array(results_instruct).mean(),
    np.array(results_nemotron).mean(),
    np.array(results_hermes).mean(),
    np.array(results_reflection).mean()
]

sems = [
    np.array(results_centaur).std() / math.sqrt(len(results_centaur)),
    np.array(results_llama).std() / math.sqrt(len(results_centaur)),
    #np.array(results_instruct).std() / math.sqrt(len(results_centaur)),
    np.array(results_nemotron).std() / math.sqrt(len(results_centaur)),
    np.array(results_hermes).std() / math.sqrt(len(results_centaur)),
    np.array(results_reflection).std() / math.sqrt(len(results_centaur))
]
ax.bar(np.arange(5), means, yerr=sems, color=['#69005f', '#ff506e', 'C0', 'C1', 'C2'])
ax.set_xticks(np.arange(5), ['Centaur', 'Llama', 'Nemotron', 'Hermes', 'Reflection'])

ax.set_ylabel('Negative log-likelihood')
ax.containers[1][0].set_alpha(0.8)
ax.containers[1][1].set_alpha(0.8)
ax.containers[1][2].set_alpha(0.8)
ax.containers[1][3].set_alpha(0.8)
ax.containers[1][4].set_alpha(0.8)
ax.set_ylim(0.9  * min(means), 1.1 * max(means))

sns.despine()
plt.tight_layout()
plt.savefig('figures/fig11.pdf', bbox_inches='tight')
plt.show()

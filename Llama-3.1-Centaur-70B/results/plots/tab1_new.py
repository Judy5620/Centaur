import pandas as pd
import math
from functools import reduce
import torch
import numpy as np 

df_exp = pd.read_csv('../experiments.csv', sep=';')
print(df_exp)
centaur_70b = torch.load('../results/custom_metrics_full_log_likelihoods_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth')
llama_70b = torch.load('../results/custom_metrics_full_log_likelihoods_unsloth-Meta-Llama-3.1-70B-bnb-4bit.pth')
baselines_full = torch.load('../results/custom_metrics_full_log_likelihoods_baselines.pth')

papers = []
ll_centaur = {}
ll_llama = {}
ll_baseline = {}

for key in centaur_70b.keys():
    print(key)
    exp_name = df_exp[df_exp['path'] == key + '/']['task_name'].item()
    if exp_name in ll_centaur.keys():
        ll_centaur[exp_name] = np.concatenate((ll_centaur[exp_name], centaur_70b[key]))
    else:
        ll_centaur[exp_name] = centaur_70b[key]

    if exp_name in ll_llama.keys():
        ll_llama[exp_name] = np.concatenate((ll_llama[exp_name], llama_70b[key]))
    else:
        ll_llama[exp_name] = llama_70b[key]
        
    if key in baselines_full.keys():
        if exp_name in ll_baseline.keys():
            ll_llama[exp_name] = np.concatenate((ll_baseline[exp_name], baselines_full[key]))
        else:
            ll_baseline[exp_name] = baselines_full[key]

print(papers)

prompt = '\\begin{table}[]\n'
prompt += '\\centering \n'
prompt += '\\begin{tabular}{@{}lccc@{}} \n'
prompt += '\\toprule  \n'
prompt += "\\textbf{Experiment} & \\textbf{Centaur} & \\textbf{Llama} & \\textbf{Cognitive model} \\\\ \n"
prompt += '\\midrule \n'
for key in  ll_centaur.keys():
    baseline_to_nan = ll_baseline[key].mean() if key in ll_baseline else np.nan
    prompt += str(key) + ' & ' + str(format(ll_centaur[key].mean(), '.4f')) + ' & ' + str(format(ll_llama[key].mean(), '.4f')) + ' & ' + str(format(baseline_to_nan, '.4f')) + ' \\\\ \n'
prompt += '\\bottomrule \\\\ \n'
prompt += '\\end{tabular} \n'
prompt += '\\caption{Full negative log-likelihoods results on held-out participants.}\n'
prompt += '\\label{tab:tab2} \n'
prompt += '\\end{table}'
print(prompt)

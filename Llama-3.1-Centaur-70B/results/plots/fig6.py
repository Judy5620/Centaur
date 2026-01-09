import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D

def get_performance(file_name):
    with open(file_name) as f:
        d = json.load(f)
        k = d.keys()
        v = [d[key]['exact_match,strict-match'] if (key == 'metabench_gsm8k') else d[key]['acc,none'] for key in k]
        verr = [d[key]['exact_match_stderr,strict-match'] if (key == 'metabench_gsm8k') else d[key]['acc_stderr,none'] for key in k]
    return k, v, verr

k, v_llama, verr_llama = get_performance('../results/metabench/base-llama-3_1-70B-results.json')
_, v_centaur, verr_centaur = get_performance('../results/metabench/centaur-2000-results.json')

df = pd.DataFrame(
    {'Task': k,
     'Llama': v_llama,
     'Centaur': v_centaur
    }).set_index('Task')


df_ci = pd.DataFrame(
    {'Task': k,
     'Llama': verr_llama,
     'Centaur': verr_centaur
    }).set_index('Task')

df.index = df.index.str.replace('metabench_arc', 'ARC')
df.index = df.index.str.replace('metabench_gsm8k', 'GSM8K')
df.index = df.index.str.replace('metabench_hellaswag', 'HellaSwag')
df.index = df.index.str.replace('metabench_mmlu', 'MMLU')
df.index = df.index.str.replace('metabench_truthfulqa', 'TruthfulQA')
df.index = df.index.str.replace('metabench_winogrande', 'Winogrande')
df.index = df.index.str.replace('metabench', 'Mean')
df_ci.index = df_ci.index.str.replace('metabench_arc', 'ARC')
df_ci.index = df_ci.index.str.replace('metabench_gsm8k', 'GSM8K')
df_ci.index = df_ci.index.str.replace('metabench_hellaswag', 'HellaSwag')
df_ci.index = df_ci.index.str.replace('metabench_mmlu', 'MMLU')
df_ci.index = df_ci.index.str.replace('metabench_truthfulqa', 'TruthfulQA')
df_ci.index = df_ci.index.str.replace('metabench_winogrande', 'Winogrande')
df_ci.index = df_ci.index.str.replace('metabench', 'Mean')
df = df[['Centaur', 'Llama']]
df_ci = df_ci[['Centaur', 'Llama']]

plt.style.use(['nature'])
plt.rcParams["figure.figsize"] = (7.08661, 3)

df.plot(kind='bar', yerr=df_ci, legend=False, color=['#69005f', '#ff506e'], alpha=0.8)
color_1 = '#69005f'
color_2 = '#ff506e'
custom_lines_r2 = [Line2D([0], [0], color=color_1, alpha=0.8, marker="o", linestyle='None', markersize=5), Line2D([0], [0], color=color_2, alpha=0.8, marker="o", linestyle='None', markersize=5)]
plt.legend(custom_lines_r2, ['Centaur', 'Llama'], frameon=False, ncols=3, bbox_to_anchor=(0.5, 1.3), loc='upper center')
plt.ylabel('Performance')
plt.xlabel('')
plt.ylim(-0.0, 1.1)

plt.tight_layout()
sns.despine()
plt.savefig('figures/fig6.pdf', bbox_inches='tight')
plt.show()

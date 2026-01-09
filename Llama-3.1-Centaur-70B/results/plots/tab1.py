import pandas as pd
import math
from functools import reduce

df_exp = pd.read_csv('../experiments.csv', sep=';')

df_llama_70b = pd.read_csv('../results/all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv')
df_llama_70b = df_llama_70b[df_llama_70b['unseen'] == 'participants'][['task', 'unsloth/Meta-Llama-3.1-70B-bnb-4bit']]

df_centaur_70b = pd.read_csv('../results/all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df_centaur_70b = df_centaur_70b[df_centaur_70b['unseen'] == 'participants'][['task', 'marcelbinz/Llama-3.1-Centaur-70B-adapter']]

df_baseline = pd.read_csv('../results/all_data_baseline.csv')
df_baseline = df_baseline[df_baseline['unseen'] == 'participants'][['task', 'baseline']]

df_random = pd.read_csv('../results/all_data_random.csv')
df_random = df_random[df_random['unseen'] == 'participants'][['task', 'random']]

df = reduce(lambda left,right: pd.merge(left,right,on=['task'], how='outer'), [df_llama_70b, df_centaur_70b, df_baseline, df_random])

for index, row in df.iterrows():
    task_name = df_exp[df_exp['path'] == df.iloc[index]['task'] + '/']['task_name'].item()
    df.loc[index, 'task'] = task_name
df = df.groupby('task', as_index=False, sort=False).mean()

df = df.rename(columns={"baseline": "Cognitive model"})
df = df.rename(columns={"marcelbinz/Llama-3.1-Centaur-70B-adapter": "Centaur"})
df = df.rename(columns={"unsloth/Meta-Llama-3.1-70B-bnb-4bit": "Llama"})
df = df.rename(columns={"random": "Random"})
df = df.rename(columns={"task": "task_name"})

print(df)

prompt = '\\begin{table}[]\n'
prompt += '\\centering \n'
prompt += '\\begin{tabular}{@{}lccc@{}} \n'
prompt += '\\toprule  \n'
prompt += "\\textbf{Experiment} & \\textbf{Centaur} & \\textbf{Llama} & \\textbf{Cognitive model} \\\\ \n"
prompt += '\\midrule \n'
for i, row in  df.iterrows():
    prompt += str(row['task_name']) + ' & ' + str(format(row['Centaur'], '.4f')) + ' & ' + str(format(row['Llama'], '.4f')) + ' & ' + str(format(row['Cognitive model'], '.4f')) + ' \\\\ \n'
prompt += '\\bottomrule \\\\ \n'
prompt += '\\end{tabular} \n'
prompt += '\\caption{Full negative log-likelihoods results on held-out participants.}\n'
prompt += '\\label{tab:tab2} \n'
prompt += '\\end{table}'
print(prompt)

df['Centaur'] = 1 - (-df['Centaur']/-df['Random'])
df['Llama'] = 1 - (-df['Llama']/-df['Random'])
df['Cognitive model'] = 1 - (-df['Cognitive model']/-df['Random'])

prompt = '\\begin{table}[]\n'
prompt += '\\centering \n'
prompt += '\\begin{tabular}{@{}lccc@{}} \n'
prompt += '\\toprule  \n'
prompt += "\\textbf{Experiment} & \\textbf{Centaur} & \\textbf{Llama} & \\textbf{Cognitive model} \\\\ \n"
prompt += '\\midrule \n'
for i, row in  df.iterrows():
    prompt += str(row['task_name']) + ' & ' + str(format(row['Centaur'], '.4f')) + ' & ' + str(format(row['Llama'], '.4f')) + ' & ' + str(format(row['Cognitive model'], '.4f')) + ' \\\\ \n'
prompt += '\\bottomrule \\\\ \n'
prompt += '\\end{tabular} \n'
prompt += '\\caption{Full pseudo-R$^2$ results on held-out participants.}\n'
prompt += '\\label{tab:tab3} \n'
prompt += '\\end{table}'

print(prompt)

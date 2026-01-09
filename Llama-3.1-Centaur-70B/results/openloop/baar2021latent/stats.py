import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('gameDat.csv')
df['correct'] = df['CorrAns'] == df['GivenAns']

df_nat = df[df['Variant'] == 'nat']
df_inv = df[df['Variant'] == 'inv']

print(df_nat['correct'].mean())
print(df_inv['correct'].mean())

print(df_nat.groupby('subID')['correct'].mean().values)
print(df_inv.groupby('subID')['correct'].mean().values)

df = pd.read_csv('simulation_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv')
df['correct'] = df['CorrAns'] == df['GivenAns']
df = df.groupby('subID').filter(lambda x: ~((x.GivenAns == 'coop').all()))
df = df.groupby('subID').filter(lambda x: ~((x.GivenAns == 'def').all()))

df_nat = df[df['Variant'] == 'nat']
df_inv = df[df['Variant'] == 'inv']

print(df_nat['correct'].mean())
print(df_inv['correct'].mean())

print(df_nat.groupby('subID')['correct'].mean().values)
print(df_inv.groupby('subID')['correct'].mean().values)

sns.swarmplot(data=[df_nat.groupby('subID')['correct'].mean().values * 100, df_inv.groupby('subID')['correct'].mean().values * 100])
ax = sns.violinplot(data=[df_nat.groupby('subID')['correct'].mean().values * 100, df_inv.groupby('subID')['correct'].mean().values * 100])
plt.setp(ax.collections, alpha=.5)
plt.xticks([0, 1], ['Human\nstrategies', 'Artificial\nstrategies'])
plt.axhline(y=50, color='grey', linestyle='--', linewidth=1.0)
plt.ylabel('Accuracy (%)')
sns.despine()
plt.show()

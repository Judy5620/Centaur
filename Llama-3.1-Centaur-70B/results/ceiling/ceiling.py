from models import NoiseCeiling, DunningKruger
from trainers import Trainer
import pandas as pd
import torch
from datasets import load_dataset
import os
import numpy as np


experiments = [
    {'prefix': '', 'path': 'ruggeri2022globalizability/exp1.csv', 'model': NoiseCeiling(UID='question')},
    {'prefix': '', 'path': 'peterson2021using/exp1.csv', 'model': NoiseCeiling(UID='uniqueID')},
    {'prefix': '../openloop/', 'path': 'jansen2021dunningkruger/exp1.csv', 'model': DunningKruger()},

]

data = []
for index in range(len(experiments)):
    exp_name = experiments[index]['path']
    prefix = experiments[index]['prefix']
    print(prefix + exp_name)

    df = pd.read_csv(prefix + exp_name)

    train_dataset = load_dataset("marcelbinz/Psych-101")['train'].filter(lambda example: example['experiment'].startswith(exp_name))
    eval_dataset = load_dataset("marcelbinz/Psych-101-test")['test'].filter(lambda example: example['experiment'].startswith(exp_name))

    train_participants = list(map(int, train_dataset['participant']))
    eval_participants = list(map(int, eval_dataset['participant']))

    train_df = df[df['participant'].isin(train_participants)]
    eval_df = df[df['participant'].isin(eval_participants)]

    trainer = Trainer(experiments[index]['model'])
    predictive_nll = trainer.fit_and_evaluate(train_df, eval_df).item()

    print(predictive_nll)

    x = exp_name.split("/")
    data.append([x[-2], x[-1], predictive_nll])

df = pd.DataFrame(data, columns=['task', 'exp', 'nll'])
print(df)
df.to_csv('results/ceiling.csv')

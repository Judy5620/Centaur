from models import DunningKruger, RescorlaWagnerModel, DualSystemsModel
from trainers import Trainer
import pandas as pd
import torch
import os
import numpy as np

experiments = [
    {'path': 'feher2020humans/exp1.csv', 'model': DualSystemsModel},
    {'path': 'dubois2022value/exp1.csv', 'model': RescorlaWagnerModel},
    {'path': 'jansen2021logic/exp1.csv', 'model': DunningKruger},
]

data = []
for index in range(len(experiments)):
    exp_name = experiments[index]['path']
    print(exp_name)

    df = pd.read_csv(exp_name)

    num_splits = 10
    splits = np.array_split(df['participant'].unique(), num_splits)

    predictive_nll = 0
    for split in splits:
        train_df = df[~df['participant'].isin(split.tolist())]
        eval_df = df[df['participant'].isin(split.tolist())]

        trainer = Trainer(experiments[index]['model']())
        predictive_nll += trainer.fit_and_evaluate(train_df, eval_df).item()

    predictive_nll = predictive_nll / num_splits

    print(predictive_nll)

    x = exp_name.split("/")
    data.append([x[-2], x[-1], predictive_nll])

df = pd.DataFrame(data, columns=['task', 'exp', 'nll'])
print(df)
df.to_csv('results/privileged.csv')

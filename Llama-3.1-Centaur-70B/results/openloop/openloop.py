from models import RescorlaWagnerModel, DualSystems
from trainers import Trainer
import pandas as pd
import torch
from datasets import load_dataset
import os

experiments = [
    #{'name': 'horizon1', 'agent': 'centaur', 'path': 'wilson2014humans/simulation0.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon1', 'agent': 'human', 'path': 'wilson2014humans/exp1.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon2', 'agent': 'centaur', 'path': 'wilson2014humans/simulation2.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon2', 'agent': 'human', 'path': 'wilson2014humans/exp3.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon3', 'agent': 'centaur', 'path': 'wilson2014humans/simulation3.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon3', 'agent': 'human', 'path': 'wilson2014humans/exp4.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon4', 'agent': 'centaur', 'path': 'wilson2014humans/simulation4.csv', 'model': RescorlaWagnerModel(num_options=2)},
    #{'name': 'horizon4', 'agent': 'human', 'path': 'wilson2014humans/exp5.csv', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'twostep1', 'agent': 'centaur', 'path': 'kool2016when/simulation.csv', 'model': DualSystems(variant='two_step')},
    {'name': 'twostep1', 'agent': 'human', 'path': 'kool2016when/exp2.csv', 'model': DualSystems(variant='two_step')},
    {'name': 'twostep2', 'agent': 'centaur', 'path': 'kool2017cost/simulation.csv', 'model': DualSystems(variant='two_step')},
    {'name': 'twostep2', 'agent': 'human', 'path': 'kool2017cost/exp2.csv', 'model': DualSystems(variant='two_step')},
]

for index in range(len(experiments)):
    data = []

    df = pd.read_csv(experiments[index]['path'])

    # select human participants
    if (('twostep' in experiments[index]['name']) and (experiments[index]['agent'] == 'human')) or (('horizon' in experiments[index]['name']) and (experiments[index]['agent'] == 'human')):
        dataset = load_dataset("marcelbinz/Psych-101-test")
        eval_dataset = dataset['test'].filter(lambda example: example['experiment'].startswith(experiments[index]['path']))
        eval_participants = list(map(int, eval_dataset['participant']))
        df = df[df['participant'].isin(eval_participants)]
        print(eval_participants)

    # match simulated data
    if ('horizon' in experiments[index]['name']):
        df = df[df['participant'] < 100]
        df = df[df['task'] < 100]

    for participant in df['participant'].unique():
        df_participant = df[df['participant'] == participant]

        trainer = Trainer(experiments[index]['model'])
        nll = trainer.fit_and_evaluate(df_participant, df_participant).item()

        if ('horizon' in experiments[index]['name']):
            params = trainer.model.information_logits.beta.item()
            data.append([participant, params, df_participant[df_participant['forced'] == 0]['reward'].mean()])
        elif ('twostep' in experiments[index]['name']):
            params = torch.sigmoid(trainer.model.tau).item()
            data.append([participant, params, df_participant['reward'].mean()])

    df = pd.DataFrame(data, columns=['participant', 'param', 'reward'])
    print(df)
    df.to_csv('results/baselines_openloop_' +  experiments[index]['agent'] + '_' + experiments[index]['name'] + '.csv')

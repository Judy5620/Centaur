import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from unsloth import FastLanguageModel
import transformers
from datasets import load_dataset
import math
import argparse

def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)

def generate_rewards(values):
    return np.round(np.clip(np.random.normal(values, 8.0), 1.0, 100.0)).astype('int')

def generate_prompts_horizon(datasets, model):

    model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = model,
      max_seq_length = 32768,
      dtype = None,
      load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    pipe = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                trust_remote_code=True,
                pad_token_id=0,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=1,
    )

    for dataset_idx, dataset in enumerate(datasets):
        print(dataset)
        df = pd.read_csv(dataset)
        prompts = load_dataset("marcelbinz/Psych-101-test")
        eval_prompts = prompts['test'].filter(lambda example: example['experiment'].startswith('wilson2014humans/' + dataset))
        eval_participants = list(map(int, eval_prompts['participant']))
        df = df[df['participant'].isin(eval_participants)]
        print(len(df.participant.unique()), flush=True)

        data = []

        for participant in tqdm(df.participant.unique()):
            choice_options = randomized_choice_options(num_choices=2)

            prompt = \
                "You are participating in multiple games involving two slot machines, labeled " + choice_options[0] + " and " + choice_options[1] + ".\n" \
                "The two slot machines are different across different games.\nEach time you choose a slot machine, you get some points.\n" \
                "You choose a slot machine by pressing the corresponding key.\n" \
                "Each slot machine tends to pay out about the same amount of points on average.\n" \
                "Your goal is to choose the slot machines that will give you the most points across the experiment.\n" \
                "The first 4 trials in each game are instructed trials where you will be told which slot machine to choose.\n" \
                "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials.\n\n"

            df_participant = df[(df['participant'] == participant)]
            num_tasks = min(100, df_participant.task.max() + 1) #

            for task in range(num_tasks):
                df_task = df_participant[(df_participant['task'] == task)]
                num_trials = int(df_task.trial.max() + 1)
                prompt += f"Game {task + 1}. There are {num_trials} trials in this game.\n"

                rewards = generate_rewards(df_task[['slot1_value', 'slot2_value']].values)

                for trial in range(num_trials):
                    df_trial = df_task[(df_task['trial'] == trial)]
                    if trial < 4:
                        c_idx = df_trial.choice.item()
                        c = choice_options[c_idx].item()
                        r = rewards[trial, c_idx]
                        prompt += f"You are instructed to press {c} and get {r} points.\n"
                        forced_choice_trial = 1
                    else:
                        prompt += f"You press <<"
                        c = pipe(prompt)[0]['generated_text'][len(prompt):]
                        if c not in [choice_options[0], choice_options[1]]:
                            c = np.random.choice([choice_options[0], choice_options[1]])
                            print('should not happen!', flush=True)
                        c_idx = list(choice_options).index(c)
                        r = rewards[trial, c_idx]
                        prompt += f"{c}>> and get {r} points.\n"
                        forced_choice_trial = 0

                    #print(prompt)
                    #print()
                    row = [participant, task, trial, c_idx, r, forced_choice_trial, df_trial['slot1_value'].item(), df_trial['slot2_value'].item()]
                    data.append(row)

                prompt += '\n'

        df = pd.DataFrame(data, columns=['participant', 'task', 'trial', 'choice', 'reward', 'forced', 'slot1_value', 'slot2_value'])
        print(df)
        df.to_csv('simulation' + str(dataset_idx) + '.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    files = os.listdir(".")
    datasets = sorted([f for f in files if (f.startswith("exp") and f.endswith(".csv"))])
    print(datasets)
    generate_prompts_horizon(datasets, args.model)

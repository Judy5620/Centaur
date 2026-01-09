import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
import transformers
from datasets import load_dataset
import os
import argparse

def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv("exp2.csv")

    dataset = load_dataset("marcelbinz/Psych-101-test")

    kool_eval_dataset = dataset['test'].filter(lambda example: example['experiment'].startswith('kool2017cost/exp2.csv'))
    kool_eval_participants = list(map(int, kool_eval_dataset['participant']))
    df = df[df['participant'].isin(kool_eval_participants)]
    print(len(df.participant.unique()))

    model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = args.model,
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

    instructions = (
        "You will be taking one of the spaceships {spaceship_0} or {spaceship_1} to one of the planets {planet_0} or {planet_1}.\n"
        "The spaceships can fly to either planet, but one will mostly fly to planet {planet_0}, and the other will mostly fly to planet {planet_1}.\n"
        "The planet a spaceship goes to most won't change during the game.\n"
        "Planet {planet_0} has aliens {alien_0} and {alien_1}, and planet {planet_1} has aliens {alien_2} and {alien_3} on it.\n"
        "Each alien has its own space treasure mine.\n"
        "When you arrive at each planet, you will ask one of the aliens for space treasure from their mines.\n"
        "The treasure an alien can give will change slowly during the game.\n"
        "Before you choose a spaceship, you will be told whether there is a treasure multiplier.\n"
        "If there is a treasure multiplier and you find treasure, you will receive 5 treasure pieces.\n"
        "If there is no treasure multiplier and you find treasure, you will receive 1 treasure piece.\n"
        "You can take a spaceship or ask an alien for space treasure by pressing the corresponding key.\n"
        "Your goal is to get as much treasure as possible over the course of the next {n_trials} days.\n\n"
    )

    data = []

    for participant in tqdm(df.participant.unique()):
        (
            spaceship_0,
            spaceship_1,
            planet_0,
            planet_1,
            alien_0,
            alien_1,
            alien_2,
            alien_3,
        ) = randomized_choice_options(8)

        par_text = instructions.format(
            spaceship_0=spaceship_0,
            spaceship_1=spaceship_1,
            planet_0=planet_0,
            planet_1=planet_1,
            alien_0=alien_0,
            alien_1=alien_1,
            alien_2=alien_2,
            alien_3=alien_3,
            n_trials=int(
                df[df.participant == participant].trial.nunique() / 2
            ),  # because two step
        )

        par_df = df[df.participant == participant].reset_index(drop=True)

        for trial in range(0, par_df.trial.nunique(), 2):
            # select the current two trials
            # by row number
            first_step_df = par_df.iloc[trial, :].copy()

            par_text += (
                "There is no treasure multiplier."
                if first_step_df.stake_level.item() == 1
                else "There is a treasure multiplier."
            )

            par_text += f" You are presented with spaceships {spaceship_0} and {spaceship_1}."

            par_text += " You press <<"

            choice_1 = pipe(par_text)[0]['generated_text'][len(par_text):]
            print(choice_1)
            if choice_1 not in [spaceship_0, spaceship_1]:
                choice_1 = np.random.choice([spaceship_0, spaceship_1])
                print('should not happen!', flush=True)

            if choice_1 == spaceship_0:
                planet_landed = np.random.choice([planet_0, planet_1], p=[0.7, 0.3])
                choice_1_idx = 0
            elif choice_1 == spaceship_1:
                planet_landed = np.random.choice([planet_1, planet_0], p=[0.7, 0.3])
                choice_1_idx = 1

            if planet_landed == planet_0:
                par_text += (
                    f"{choice_1}>>."
                    f" You end up on planet {planet_landed}."
                    f" You see alien {alien_0} and alien {alien_1}."
                )
            elif planet_landed == planet_1:
                par_text += (
                    f"{choice_1}>>."
                    f" You end up on planet {planet_landed}."
                    f" You see alien {alien_2} and alien {alien_3}."
                )

            par_text += " You press <<"

            choice_2 = pipe(par_text)[0]['generated_text'][len(par_text):]
            print(choice_2)
            avail_options = [alien_0, alien_1] if planet_landed == planet_0 else [alien_2, alien_3]
            if choice_2 not in avail_options:
                choice_2 = np.random.choice(avail_options)
                print('should not happen!', flush=True)

            if choice_2 == alien_0:
                reward = np.random.choice([1, 0], p=[first_step_df['reward.0.0'].item(), 1 - first_step_df['reward.0.0'].item()])
                state_idx = 0
                choice_2_idx = 0
            elif choice_2 == alien_1:
                reward = np.random.choice([1, 0], p=[first_step_df['reward.0.1'].item(), 1 - first_step_df['reward.0.1'].item()])
                state_idx = 0
                choice_2_idx = 1
            elif choice_2 == alien_2:
                reward = np.random.choice([1, 0], p=[first_step_df['reward.1.0'].item(), 1 - first_step_df['reward.1.0'].item()])
                state_idx = 1
                choice_2_idx = 0
            elif choice_2 == alien_3:
                reward = np.random.choice([1, 0], p=[first_step_df['reward.1.1'].item(), 1 - first_step_df['reward.1.1'].item()])
                state_idx = 1
                choice_2_idx = 1

            par_text += (
                f"{choice_2}>>."
                f" You find {int(reward)} pieces of space treasure.\n"
            )

            row1 = [participant, 0, trial, 999, choice_1_idx, 0]
            row2 = [participant, 0, trial+1, state_idx, choice_2_idx, reward]

            data.append(row1)
            data.append(row2)
            #print(par_text)
            #print()
    df = pd.DataFrame(data, columns=['participant', 'task', 'trial', 'current_state', 'choice', 'reward'])
    print(df)
    df.to_csv('simulation.csv')

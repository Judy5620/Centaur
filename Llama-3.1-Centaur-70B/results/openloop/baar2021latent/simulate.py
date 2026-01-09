import pandas as pd
import numpy as np
import jsonlines
from unsloth import FastLanguageModel
import transformers
import argparse

def randomized_choice_options(num_choices):
    choice_options = list(map(chr, range(65, 91)))
    return np.random.choice(choice_options, num_choices, replace=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

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

    df = pd.read_csv('gameDat.csv')

    data = []
    for subject in df.subID.unique():
        choice_options = randomized_choice_options(2)

        prompt = "You will take part in a Social Prediction Game.\n"\
            "You will observe a Player playing against an Opponent in different games.\n"\
            "In each game, the Player and the Opponent simultaneously choose between option " + choice_options[0] + " and option " + choice_options[1] + ".\n"\
            "The Player and the Opponent win points based on their choices.\n"\
            "The rules change between games, and you will be informed about them before each game.\n"\
            "The Player varies between blocks but is consistent across games within a block.\n"\
            "The Opponent switches in each game.\n"\
            "Your task is to predict the choices made by the Player and rate your confidence in this prediction on an 11-point scale from 0 to 100 (in increments of 10).\n"\
            "You get feedback after each game on whether your prediction was correct or not.\n\n"\

        df_sub = df[df['subID'] == subject]
        for block in range(4):
            prompt += 'Block ' + str(block + 1) + ' starts now.\n\n'
            df_block = df_sub[df_sub['Block'] == block]
            for trial in range(16):
                df_trial = df_block[df_block['Trial'] == trial]

                # 0 co-operate, 1 defect
                T = df_trial['T'].item()
                S = df_trial['S'].item()

                prompt += "The rules of the game are as follows:\n"\
                    "If Player chooses option " + choice_options[0] + " and Opponent chooses option " + choice_options[0] + ", then Player wins 10 points and Opponent wins 10 points.\n"\
                    "If Player chooses option " + choice_options[0] + " and Opponent chooses option " + choice_options[1] + ", then Player wins " + str(S) + " points and Opponent wins " + str(T) + " points.\n"\
                    "If Player chooses option " + choice_options[1] + " and Opponent chooses option " + choice_options[0] + ", then Player wins " + str(T) + " points and Opponent wins " + str(S) + " points.\n"\
                    "If Player chooses option " + choice_options[1] + " and Opponent chooses option " + choice_options[1] + ", then Player wins 5 points and Opponent wins 5 points.\n"\
                    "You predict that Player will choose option <<"
                
                # simulate choice
                choice = pipe(prompt)[0]['generated_text'][len(prompt):]
                print(choice)
                if choice == choice_options[0]:
                    response = 'coop'
                elif choice == choice_options[1]:
                    response = 'def'
                else:
                    response = 'NaN'
                    print('something went wrong')
                correct = 'correct' if response ==  df_trial['CorrAns'].item() else 'incorrect'
                prompt += str(choice) + ">>. You indicate a confidence of <<" 

                # simulate confidence
                confidence = pipe(prompt)[0]['generated_text'][len(prompt):]
                prompt += str(confidence) + ">>. Your prediction was " + correct + ".\n\n"
                print(prompt)
      
                row = [subject, block, df_trial['Player'].item(), df_trial['Type'].item(), df_trial['Variant'].item(), df_trial['Type_Total'].item(), trial, S, T, df_trial['GameType'].item(), df_trial['CorrAns'].item(), response, confidence]
                data.append(row)

    df = pd.DataFrame(data, columns=['subID', 'Block', 'Player', 'Type', 'Variant', 'Type_Total', 'Trial', 'S', 'T', 'GameType', 'CorrAns', 'GivenAns', 'ConfidenceNum'])
    print(df)
    df.to_csv('simulation_' + args.model.replace('/', '-') +  '.csv')

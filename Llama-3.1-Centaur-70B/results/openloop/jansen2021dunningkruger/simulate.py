""" Questions are reproduced from the Qualtrics file https://osf.io/e3wkr """
import pandas as pd
import random
import numpy as np
import re
from tqdm import tqdm
import sys
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
    )

    r_id = tokenizer(">>").input_ids[-1]
    print(r_id)

    df = pd.read_csv("exp1.csv")
    q = ['Q' + str(i) for i in range(1, 21)]
    questions = {
        'Q1': {'text': 'Q1. The school-age child faces a formidable task when during the first few years of classroom experiences [he or she is expected to master the printed form of language.]',
               'answers': {'he or she expects to master the printed form of language.': '',
                'he or she is expected to master the printed form of language.': '',
                'he or she faces expectations of mastering the printed form of language.': '',
                'mastery of the printed form of language is expected of him or her.': '',
                'mastery of print is expected by his or her teacher.': '',}
                },
        'Q2': {'text': 'Q2. He came to the United States as a young [man, he found] a job as a coal miner.',
               'answers': {'man, he found': '',
               'man and found': '',
               'man and there he was able to find': '',
               'man and then finding': '',
               'man and had found': '',},
                },
        'Q3': {'text': 'Q3. To a large degree, [poetry, along with all the other arts, is] a form of imitation.',
               'answers': {'poetry, along with all the other arts, is': '',
               'poetry along with all the other arts is': '',
               'poetry, along with all the other arts, are': '',
               'poetry, and other arts, is': '',
               'poetry and art are': '',},
                },
        'Q4': {'text': 'Q4. Delegates to the political convention found [difficulty to choose] a candidate from among the few nominated.',
               'answers': {'difficulty to choose': '',
               'it difficult in making the choice of': '',
               'it difficult to choose': '',
               'choosing difficult when selecting': '',
               'making a choice difficult in selecting': "",},
                },
        'Q5': {'text': "Q5. Reading in any language can be viewed as a developmental task much the same as learning to walk, to cross the street independently, to care for one's possessions, or [accepting responsibility for one's own decisions.]",
               'answers': {"accepting responsibility for one's own decisions.": "",
               "accepting one's own decisions responsibly.": "",
               "to accept responsibility for one's own decisions.": "",
               "accepting responsibility and making one's own decisions.": "",
               "to make one's own decisions.": "",},
                },
        'Q6': {'text': 'Q6. Sea forests of giant kelp, which fringe only one coastline in the Northern Hemisphere, [is native to shores] throughout the Southern Hemisphere.',
               'answers': {'is native to shores': '',
               'is native to most shores': '',
               'are native only in shores': '',
               'are native': '',
               'are native to shores': '',},
                },
        'Q7': {'text': 'Q7. Taking an occasional respite between chapters or assignments is more desirable [than a long, continuous period of study.',
               'answers': {'than a long, continuous period of study.': '',
               'than a period of long, continuous study.': '',
               'than a long period of continuous study.': '',
               'than studying for a long, continuous period.': '',
               'than a study period long and continuous.': '',},
                },
        'Q8': {'text': 'Q8. Like so many characters in Russian fiction, [Crime and Punishment exhibits] a behavior so foreign to the American temperament that many readers find the story rather incredible.',
               'answers': {'Crime and Punishment exhibits': '',
               'those in Crime and Punishment exhibit': '',
               'those in Crime and Punishment exhibits': '',
               'often exhibiting': '',
               'characterized by': '',},
                },
        'Q9': {'text': 'Q9. Don Quixote provides a cross section of Spanish life, thought, and [portrays the feelings of many Spaniards] at the end of the chivalric age.',
               'answers': {'portrays the feelings of many Spaniards': '',
               'portrayal of the feelings of many Spaniards': '',
               'feelings portrayed by Spaniards': '',
               'feelings': '',
               'Spanish feelings': '',},
                },
        'Q10': {'text': 'Q10. Hamlet, Prince of Denmark thought several times of killing Claudius [and finally succeeding] in doing so.',
                'answers': {'and finally succeeding' : '',
                'that finally was successful': '',
                'finally a successful attempt': '',
                'being finally successful': '',
                'and finally succeeded': '',},
                },
        'Q11': {'text': 'Q11. The lamb [had laid on the hay beside its mother and had begun to nurse as soon as the boy had sat] the lantern on the table.',
                'answers': {'had laid on the hay beside its mother and had begun to nurse as soon as the boy had sat': '',
                'had lain on the hay beside its mother and had begun to nurse as soon as the boy had set': '',
                'had laid on the hay beside its mother and had begun to nurse as soon as the boy had set': '',
                'had lain on the hay besides its mother and had begun to nurse as soon as the boy had set': '',
                "had lain on the hay beside it's mother and had begun to nurse as soon as the boy had set": '',},
                },
        'Q12': {'text': 'Q12. An infant, [whether lying alone in the crib or enjoying the company of adults, is consistently fascinated at] the movement of toes and fingers.',
                'answers': {'whether lying alone in the crib or enjoying the company of adults, is consistently fascinated at': '',
                'alone or in company, is consistently fascinated at': '','whether lying alone in the crib or enjoying the company of adults, is constantly fascinated at': '',
                'whether lying alone in the crib or enjoying the company of adults, is consistently fascinated by': '',
                'lonely in the crib and enjoying the company of adults is consistently fascinated at': '',},
                },
        'Q13': {'text': 'Q13. A policeman of proven valor, [the city council designated him] the "Outstanding Law Enforcement Officer of the Year."',
                'answers': {'the city council designated him': '',
                "the city council's designating him": '',
                'the city council will designate him': '',
                'he designated the city council': '',
                'he was designated by the city council': '',},
                },
        'Q14': {'text': 'Q14. The supervisor asked, ["Bob have you checked with our office in Canton, Ohio, to see if it stocks slate, flagstone, and feather rock?"]',
                'answers': {'"Bob have you checked with our office in Canton, Ohio, to see if it stocks slate, flagstone, and feather rock?"': '',
                '"Bob, have you checked with our office in Canton, Ohio to see if it stocks slate, flagstone, and feather rock?"': '',
                'Bob, have you checked with our office in Canton, Ohio, to see if it stocks slate, flagstone, and feather rock?': '',
                '"Bob, have you checked with our office in Canton, Ohio, to see if it stocks slate, flagstone, and feather rock?"': '',
                '"Bob have you checked with our office in Canton, Ohio, to see if it stocks slate flagstone and feather rock?"': '',},
                },
        'Q15': {'text': 'Q15. [If the room would have been brighter], I would have been more successful in my search for the lost earrings.',
                'answers': {'If the room would have been brighter': '',
                'If rooms were brighter': '',
                'If the room could have been brighter': '',
                'If the room had been brighter': '',
                'If the room was brighter': '',},
                },
        'Q16': {'text': 'Q16. After announcing that no notes could be used during the final exam, the instructor was compelled to fail [two students because they used notes anyway.]',
                'answers': {'two students because they used notes anyway.': '',
                'two students because of their notes.': '',
                'two students because of them using notes.': '',
                'two students whose notes were used.': '',
                'two students due to the use of their notes.': '',},
                },
        'Q17': {'text': 'Q17. The respiratory membranes, [through which exchange of gases occurs], are the linings of the lungs.',
                'answers': {'through which exchange of gases occurs': '',
                'through which exchange of gas occurs': '',
                'after gases are exchanged': '',
                'occurs through the exchange of gases': '',
                'through which gas is exchanged': '',},
                },
        'Q18': {'text': 'Q18. Jeff is one of those [who tends to resist any attempt at] classification or regulation.',
                'answers': {'who tends to resist any attempt at': '',
                'whose tendency to resist any attempt at': '',
                'who tend to resist any attempt at': '',
                'who tends to resist any attempt to': '',
                'who tends to resistance of any attempt at': '',},
                },
        'Q19': {'text': 'Q19. [The amount of water in living cells vary], but it is usually 65 percent and in some organisms may be as high as 96 percent or more of the total substance.',
                'answers': {'The amount of water in living cells vary': '',
                'The amount of water varies': '',
                'The amount of water in cells vary': '',
                'The amount of water in living cells varies': '',
                'The amounts of water varies in living cells': '',},
                },
        'Q20': {'text': 'Q20. [The belief of ancient scientists was] that maggots are generated from decaying bodies and filth and are not formed by reproduction.',
                'answers': {'The belief of ancient scientists was': '',
                'The ancient scientists beliefs were': '',
                'The ancient scientists believe': '',
                'The belief of ancient scientists were': '',
                'The ancient belief of scientists was': '',},
                },
    }

    correct_answers = [
        'he or she is expected to master the printed form of language.',
        'man and found',
        'poetry, along with all the other arts, is',
        'it difficult to choose',
        "to accept responsibility for one's own decisions.",
        'are native to shores',
        'than studying for a long, continuous period.',
        'those in Crime and Punishment exhibit',
        'feelings',
        'and finally succeeded',
        'had lain on the hay beside its mother and had begun to nurse as soon as the boy had set',
        'whether lying alone in the crib or enjoying the company of adults, is consistently fascinated by',
        'he was designated by the city council',
        '"Bob, have you checked with our office in Canton, Ohio, to see if it stocks slate, flagstone, and feather rock?"',
        'If the room had been brighter',
        'two students because they used notes anyway.',
        'through which exchange of gases occurs',
        'who tend to resist any attempt at',
        'The amount of water in living cells varies',
        'The belief of ancient scientists was'
    ]

    data = []
    for participant in df.participant.unique():
        print(participant)
        if participant == 1000:
            break # TODO
        df_participant = df[(df['participant'] == participant)]
        buttons = list(randomized_choice_options(num_choices=5))
        questions_participant = questions
        for question in q:
            b = random.sample(buttons, len(buttons))
            questions_participant[question]['answers'] = {k: b.pop() for k in questions[question]['answers'].keys()}

        prompt = "You're about to answer a set of 20 questions about grammar. How many of the 20 questions do you think you will answer correctly?\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n'
        data.append([participant, 'absAssess0', choice, 0, 0])

        prompt += "Compared to other participants in this study, how well do you think you will do? Marking 90% means you will do better than 90% of participants, marking 10% means you will do better than only 10%, and marking 50% means that you will perform better than half of the participants.\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n'
        data.append([participant, 'relAssess0', choice, 0, 1])

        prompt += "On a scale of 0 to 10, how difficult is recognizing correct grammar for the average participant?\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n'
        data.append([participant, 'diffOther0', choice, 0, 2])

        prompt += "On a scale of 0 to 10, how difficult is recognizing correct grammar for you?\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n\n'
        data.append([participant, 'diffSelf0', choice, 0, 3])

        prompt += 'You will now see twenty questions.\nIn each question, some part of each sentence is in square brackets.\n'\
        'Five choices for rephrasing that part follow each sentence; one choice repeats the original, and the other four are different.\n'

        prompt += f"Your task is to use the buttons {buttons[0]}, {buttons[1]}, {buttons[2]}, {buttons[3]}, and {buttons[4]} to select the grammatically correct choice.\n\n"

        score = 0
        for question_counter, question in enumerate(q):
            prompt += f"{questions_participant[question]['text']}\nThe choices are:\n"
            for key, value in questions_participant[question]['answers'].items():
                prompt += f'{value}: {key}\n'
            prompt += f"You press <<"
            choice = pipe(prompt, max_new_tokens=1, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
            prompt += choice + ">>."
            prompt += '\n\n'
            choice_string = list(questions_participant[question]['answers'].keys())[list(questions_participant[question]['answers'].values()).index(choice)]
            if choice_string == correct_answers[question_counter]:
                score += 1
            data.append([participant, 'Q' + str(question_counter+1), choice_string, 0, 4+question_counter])

        data.append([participant, 'score', score, 0, 24])

        prompt += "How many of the 20 grammar questions you just completed do you think you answered correctly?\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n'
        data.append([participant, 'absAssess1', choice, 0, 25])

        prompt += "Compared to other participants in this study, how well do you think you performed? Marking 90% means you will do better than 90% of participants, marking 10% means you will do better than only 10%, and marking 50% means that you will perform better than half of the participants.\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n'
        data.append([participant, 'relAssess1', choice, 0, 26])

        prompt += "On a scale of 0 to 10, how difficult was recognizing correct grammar for the average participant?\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        prompt += '\n'
        data.append([participant, 'diffOther1', choice, 0, 27])

        prompt += "On a scale of 0 to 10, how difficult was recognizing correct grammar for you?\n"
        prompt += f'You say <<'
        choice = pipe(prompt, max_new_tokens=2, eos_token_id=r_id)[0]['generated_text'][len(prompt):]
        choice = re.sub("[^0-9]", "", choice)
        prompt += choice + ">>."
        data.append([participant, 'diffSelf1', choice, 0, 28])

        if participant == 0:
            print(prompt)

    df = pd.DataFrame(data, columns=['participant', 'question', 'choice', 'task', 'trial'])
    print(df)
    df.to_csv('simulation.csv')

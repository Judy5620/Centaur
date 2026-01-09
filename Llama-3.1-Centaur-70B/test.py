from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import pandas as pd
import argparse
import torch

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument("--model", type=str, required=True)
      args = parser.parse_args()

      task_names = [
            "badham2017deficits",
            "bahrami2020four",
            "enkavi2019adaptivenback",
            "enkavi2019digitspan",
            "enkavi2019gonogo",
            "enkavi2019recentprobes",
            "feng2021dynamics",
            "flesch2018comparing",
            "frey2017cct",
            "frey2017risk",
            "gershman2018deconstructing",
            "gershman2020reward",
            "hebart2023things",
            "hilbig2014generalized",
            "kool2016when",
            "kool2017cost",
            "lefebvre2017behavioural",
            "levering2020revisiting",
            "ludwig2023human",
            "peterson2021using",
            "plonsky2018when",
            "ruggeri2022globalizability",
            "sadeghiyeh2020temporal",
            "schulz2020finding",
            "somerville2017charting",
            "speekenbrink2008learning",
            "steingroever2015data",
            "tomov2020discovery",
            "tomov2021multitask",
            "waltz2020differential",
            "wilson2014humans",
            "wu2023chunking",
            "wulff2018description",
            "wulff2018sampling",
            "xiong2023neural",
            "zorowitz2023data",
      ]

      model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map="auto")
      tokenizer = AutoTokenizer.from_pretrained(args.model)
      l_id = tokenizer(" <<").input_ids[1:]
      r_id = tokenizer(">>").input_ids[1:]
      collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)
      dataset = load_dataset("marcelbinz/Psych-101-test")
      is_quantized = model.is_quantized

      data = []
      with torch.no_grad():
            for task_name in task_names:
                  eval_dataset = dataset['test'].filter(lambda example: example['experiment'].startswith(task_name))
                  
                  model.is_quantized = False
                  training_args = TrainingArguments(
                        output_dir="eval",
                        per_device_eval_batch_size=1
                  )
                  trainer = SFTTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        train_dataset=eval_dataset,
                        eval_dataset=eval_dataset,
                        dataset_text_field="text",
                        max_seq_length=32768,
                        data_collator=collator,
                  )
                  model.is_quantized = is_quantized
                  result = trainer.evaluate()

                  print(task_name, flush=True)
                  print(result, flush=True)
                  data.append([task_name, result['eval_loss']])
            df = pd.DataFrame(data, columns=['task', str(args.model)])
            print(df, flush=True)
            df.to_csv('results/' + args.model.replace('/', '-') +  '.csv')

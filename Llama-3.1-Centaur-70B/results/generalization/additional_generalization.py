from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd
import argparse
import torch

def full_log_likelihoods(logits, labels):
	with torch.no_grad():
		logits = logits.float().cpu()
		labels = labels.cpu()
		labels = torch.cat((labels[0, 1:], -100 * torch.ones(1).long()), 0)
		logits = logits[0]
		ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
		total_loss = []
		item_loss = 0
		item_counter = 0
		for i in range(ce.shape[0]):
			if labels[i] != -100:
				item_loss += ce[i]
				item_counter += 1
			else:
				if item_counter != 0:
					total_loss.append(item_loss)
					item_loss = 0
					item_counter = 0
		return torch.Tensor(total_loss)


def compute_metrics(pred):
	return {'custom_loss': pred.predictions}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True)
	args = parser.parse_args()

	task_names = [
            "additional_experiments/awad2018moral.jsonl",
            "additional_experiments/demircan2024evaluatingcategory.jsonl",
			"additional_experiments/demircan2024evaluatingreward.jsonl",
            "additional_experiments/akata2023repeatedgames.jsonl",
			"additional_experiments/singh2022representing.jsonl",
            "additional_experiments/xu2021novelty.jsonl",
      ]
	
	model, tokenizer = FastLanguageModel.from_pretrained(
	  model_name = args.model,
	  max_seq_length = 32768,
	  dtype = None,
	  load_in_4bit = True,
	)
	l_id = tokenizer(" <<").input_ids[1:]
	r_id = tokenizer(">>").input_ids[1:]
	collator = DataCollatorForCompletionOnlyLM(response_template=l_id, instruction_template=r_id, tokenizer=tokenizer)
	is_quantized = model.is_quantized

	data = {}
	with torch.no_grad():
		for i, task_name in enumerate(task_names):
			dataset = load_dataset('json',
            	data_files={
                	'test': [task_name],
                }
            )

			model.is_quantized = False
			training_args = TrainingArguments(
		        output_dir="eval_"+str(i),
                per_device_eval_batch_size=1,
                report_to="none",
				eval_accumulation_steps=1
			)
			trainer = SFTTrainer(
				model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset['test'],
                eval_dataset=dataset['test'],
                dataset_text_field="text",
				max_seq_length=32768,
				data_collator=collator,
				compute_metrics=compute_metrics,
				preprocess_logits_for_metrics=full_log_likelihoods,
			)
			model.is_quantized = is_quantized
			result = trainer.evaluate()

			print(task_name, flush=True)
			print(result, flush=True)
			data[task_name] = result['eval_custom_loss']

		torch.save(data, 'results/additional_generalization_full_log_likelihoods_' + args.model.replace('/', '-') +  '.pth')
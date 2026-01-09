from unsloth import FastLanguageModel
import transformers

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "marcelbinz/Llama-3.1-Centaur-70B-adapter",
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

prompt = "You will be presented with triplets of objects, which will be assigned to the keys H, Y, and E.\n" \
  "In each trial, please indicate which object you think is the odd one out by pressing the corresponding key.\n" \
  "In other words, please choose the object that is the least similar to the other two.\n\n" \
  "H: plant, Y: chainsaw, and E: periscope. You press <<H>>.\n" \
  "H: tostada, Y: leaf, and E: sail. You press <<H>>.\n" \
  "H: clock, Y: crystal, and E: grate. You press <<Y>>.\n" \
  "H: barbed wire, Y: kale, and E: sweater. You press <<E>>.\n" \
  "H: raccoon, Y: toothbrush, and E: ice. You press <<"

print(prompt)

choice = pipe(prompt)[0]['generated_text'][len(prompt):]
print(choice)

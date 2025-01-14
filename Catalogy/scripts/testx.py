from transformers import GPTJForCausalLM, AutoTokenizer
import torch

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
context = """Instruction: Describe the British Shorthair cat breed using the words:
Slightly distracted, Aggressive, Very vigilant, Extremely clever.
Answer: """

input_ids = tokenizer(context, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

'''from keytotext import pipeline

nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")
print(nlp(["British shorthair", "Slightly distracted", "Aggressive", "Very vigilant", "Extremely clever"]))'''
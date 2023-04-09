from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
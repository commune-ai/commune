import commune 
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# commune.print(commune.connect(**{'ip': '162.157.13.236', 'port': 9203}).server_registry())
commune.print(commune.server_registry())

# import transformers
# device = "cpu"

# tokenizer = transformers.LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(device)

# batch = tokenizer(
#     "The capital of Canada is",
#     return_tensors="pt", 
#     add_special_tokens=False
# )

# batch = {k: v.to(device) for k, v in batch.items()}
# generated = model.generate(batch["input_ids"], max_length=100)
# print(tokenizer.decode(generated[0]))
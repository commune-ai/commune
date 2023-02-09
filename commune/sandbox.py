import commune
from transformers import AutoConfig,AutoTokenizer
import streamlit as st

model_name = 'gptneox20b'
model_class = commune.get_module('model.transformer')
model = model_class(model_name=model_name, max_memory={0: '15GiB', 1: '15GiB', 2: '15GiB', 3: '15GiB', 4: '15Gi'}).serve(wait_for_termination=True)
# model.model.save_pretrained('/tmp/gpt125m')

# print(commune.launch('dataset.text.bittensor', mode='pm2'))


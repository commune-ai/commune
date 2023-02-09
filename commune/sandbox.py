import commune
from transformers import AutoConfig,AutoTokenizer
import streamlit as st

model_name = '/tmp/gpt125m'
model_class = commune.get_module('model.transformer')
model = model_class(model_name=model_name)
# model.model.save_pretrained('/tmp/gpt125m')

# print(commune.launch('dataset.text.bittensor', mode='pm2'))


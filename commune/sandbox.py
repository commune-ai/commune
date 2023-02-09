# import commune
# from transformers import AutoConfig,AutoTokenizer
# import streamlit as st

# model_name = 'gptneox20b'
# model_class = commune.get_module('model.transformer')
# model = model_class(model_name=model_name, max_memory={0: '15GiB', 1: '15GiB', 2: '15GiB', 3: '15GiB', 4: '15Gi'})
# model.save_pretrained('/tmp/gptneox20b')

# # print(commune.launch('dataset.text.bittensor', mode='pm2'))
import commune
from transformers import AutoConfig,AutoTokenizer
 
model = commune.connect('GPTNeoX::EleutherAI_gpt-neox-20b')

print(model.tokenize(['yo whadup']))
# print(model.forward(input_ids=' fam'))
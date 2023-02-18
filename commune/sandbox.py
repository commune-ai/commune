import commune
import streamlit as st
from typing import *

st.write(commune.servers())


from commune.utils.torch import tensor_dict_info
from commune.utils.tokenizer import decode_topk
dataset = commune.connect("dataset::bittensor")
sample = dataset.sample()

model = commune.connect('model::gptj')
output_length = 10
topk = 512
t = commune.timer()
sample['logit_remap'] = False
sample['token_remap'] = True
sample['output_length'] = 10
sample['output_logits'] = False
sample['topk'] = topk
sample['output_length'] = output_length
metrics = {}
t = commune.timer()

for m in ['gptj', 'gptjt']:
    output = model.forward(**sample)

output['logits'] = decode_topk(output['topk'])

   

metrics['seconds'] = t.seconds
metrics['topk'] = topk
metrics['output_length'] = output_length
metrics['output'] = tensor_dict_info(output)    
st.write('Info Dict')
st.write(metrics)
import torch
# decode the output


st.write('Metrics')
st.write(metrics)


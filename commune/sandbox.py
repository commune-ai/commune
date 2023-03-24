import commune
import streamlit as st
import torch
dataset = commune.connect('dataset')
model = commune.connect('model')
vec = dataset.sample(tokenize=True)['input_ids']
st.write(model.config)
input_ids = torch.clip(vec, max=model.config['model']['vocab_size']-1)

st.write(model.forward(input_ids=input_ids))
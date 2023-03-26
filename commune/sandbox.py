import commune
import streamlit as st
import torch



module = commune.get_module('model.transformer')

st.write(list(module.shortcuts.keys()))

models = [
  "gpt2.7b",
  "gpt3b",
  "gptjt",
  "gptjt_mod",
  "gptj",
  "gptj.pyg6b",
  "gptj.instruct",
  "gptj.codegen",
  "gptj.hivemind",
]

for model in models:
    model_kwargs =  {'model': model}
    if 'gptj' in model:
        model_kwargs['tokenizer'] = 'gptj'
    module.launch(name=f'model.{model}',kwargs=model_kwargs, mode='pm2')


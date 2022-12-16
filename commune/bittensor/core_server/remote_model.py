from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio 

import os, sys
sys.path.append(os.getenv('PWD'))
asyncio.set_event_loop(asyncio.new_event_loop())

import commune
import streamlit as st



class RemoteModel:
    def __init__(self, model="EleutherAI/gpt-neo-125M"):

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
st.write('bro')

st.write(RemoteModel())
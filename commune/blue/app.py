import commune as c
import json
import numpy as np
import streamlit as st

class App(c.Module):
    def __init__(self, model = 'model.openrouter', score_module='blue'):
        self.model = c.module(model)()
        self.blue_model = c.module(score_module)()

    def signin(self):
        st.write('## Sign In')
        secret = st.text_input('whats your secret ;) ? ', type='password')
        self.key = c.pwd2key(secret)
        return self.key
    
    def history(self):
        return self.get(f'history/{self.key.ss58_address}')
    

    def all_history(self):
        return self.glob('history')

    def add_history(self, text):
        return self.put(f'history/{self.key.ss58_address}', text)
    
    def model_arena(self):

        text = st.text_area('Enter your text here')
        if st.button('Submit'):
            red_response = self.model.forward(text)
            cols = st.columns(2)
            with cols[0]:
                st.write('Red Model Response')
                st.write(red_response)
            blue_response = self.blue_model.forward(red_response)
            with cols[1]:
                st.write('Blue Model Response')
                st.write(blue_response)

    def app(self):
        st.write('## Always Blue')
        with st.sidebar:
            self.signin()
        st.write('You are signed in as ' + self.key.ss58_address)

        self.model_arena()

App.run(__name__)
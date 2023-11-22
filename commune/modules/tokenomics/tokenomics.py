import commune as c
import streamlit as st
class Tokenomics(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y


    def dashboard(self):
        st.header('Tokenomics')
        st.subheader('This is a subheader')
        st.text('This is a text')

Tokenomics.run(__name__)
import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px
import streamlit as st

css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)

class SubspaceAPI(c.Module):
    
   

    def sidebar(self):
        return self.select_key()

    @classmethod
    def app(cls, backend='app'):
        while not c.server_exists(backend):
            print(f"Waiting for {backend}")
            c.serve(backend)
            c.sleep(5)
        self = cls(backend=backend)
        self.sidebar()
        self.subnets_app()

# SubspaceAPI.run(__name__)
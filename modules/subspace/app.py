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

class SubspaceDashboard(c.Module):
    
    def __init__(self, root_netuid=0, max_age = 10000, api='subspace'):
        self.state = {}
        self.max_age = max_age
        self.root_netuid = root_netuid
        self.subspace = c.module('subspace')()

    def global_state(self, max_age=None):
        global_state = self.get('global_state', None, max_age=max_age)
        if global_state == None :
            return self.subspace.global_state(max_age=max_age)
        return global_state
    
    def sync(self, netuid=None, max_age=None, update=False):
        state = self.get('state', None, max_age=max_age)
        if state == None:
            state = {}
        global_state = self.global_state(max_age=max_age)

        return self.state['modules']
    
    def select_key(self, key='module'):
        keys = c.keys()
        key2idx = {key:i for i, key in enumerate(keys )}
        key = st.selectbox("Key", keys, key2idx[key])
        self.key  = key 

        st.code(f"{self.key.ss58_address}")
        return key


    def subnets_app(self, backend='app'):
        st.title("Subnets")
        st.write(f"Connected to {backend}")
        self.sync()
        subnet_name = st.selectbox("Subnet", self.subnet_names, 0)
        with st.expander(f"{subnet_name} (netuid={netuid})"):
            st.write(self.state['params'])

        leaderboard = c.df(self.state['modules'])

        with st.expander("Leaderboard"):
            st.write(leaderboard)

    def sidebar(self):
        with st.sidebar:
            return self.select_key()

    def app(self):
        self.sidebar()
        self.subnets_app()

SubspaceDashboard.run(__name__)
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
    
    def __init__(self, root_netuid=0, max_age = 10000):
        self.max_age = max_age
        self.root_netuid = root_netuid
        self.sync_global()


    def sync_global(self):
        self.subspace = c.module('subspace')()
        self.global_params = self.subspace.global_params(max_age=self.max_age)
        self.subnet2netuid = self.subspace.subnet2netuid(max_age=self.max_age)
        self.subnet_names = list(self.subnet2netuid.keys())
        self.sync_subnet(netuid=0)
    
    def sync_subnet(self, netuid=0, update=False):
        subnet_params = self.subspace.subnet_params(netuid=netuid, max_age=self.max_age)
        subnet_modules = self.subspace.get_modules(netuid=netuid,  max_age=self.max_age)
        subnet_name = self.subnet2netuid.get(netuid)
        self.state = {
            'params': subnet_params,
            'netuid': netuid,
            'name': subnet_name,
            'modules': subnet_modules
        }
  

    def select_key(self, key='module'):
        keys = c.keys()
        key2idx = {key:i for i, key in enumerate(keys )}
        key = st.selectbox("Key", keys, key2idx[key])
        self.key  = key 

        st.code(f"{self.key.ss58_address}")
        return key


    def subnets_app(self):
        st.title("Subnets")
        self.sync_global()
        subnet_name = st.selectbox("Subnet", self.subnet_names, 0)
        netuid = self.subnet2netuid.get(subnet_name)
        self.sync_subnet(netuid=netuid)
        with st.expander(f"{subnet_name} (netuid={netuid})"):
            st.write(self.state['params'])

        leaderboard = c.df(self.state['modules'])

        with st.expander("Leaderboard"):
            st.write(leaderboard)

    def sidebar(self):
        return self.select_key()

    def app(self):
        self.sidebar()
        self.subnets_app()

SubspaceDashboard().app()
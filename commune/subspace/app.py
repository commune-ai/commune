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
    
    def __init__(self, root_netuid=0, max_age = 10000, api='subspace.api'):
        self.max_age = max_age
        self.root_netuid = root_netuid
        if not c.server_exists(api):
            c.serve(api, wait_for_server=True)
        self.api = c.connect(api)

    def global_state(self, max_age=None):
        global_state = self.get('global_state', None, max_age=max_age)
        if global_state == None :
            return self.api.global_state(max_age=max_age)
        return global_state
    
    def sync(self, max_age=None):
        global_state = self.global_state(max_age=max_age)
        self.__dict__.update(global_state)
        return global_state


    def subnet_state(self, netuid=0, max_age=None):
        subnet_state = self.get(f'subnet_state/{netuid}', None, netuid=netuid, max_age=max_age)
        if subnet_state == None:
            subnet_state = self.api.subnet_state(netuid=netuid, max_age=max_age)
        return subnet_state

    def sync_loop(self, max_age=1000, timeout=60, sleep=10):
        while True:
            self.global_state(**{'max_age': max_age})
            for netuid in self.netuids:
                c.print(f"Syncing {netuid}")
                self.subnet_state(**{'netuid': netuid, 'max_age': max_age})
                
            c.sleep(sleep)
  

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

    def app(cls):
        self = cls()
        self.sidebar()
        self.subnets_app()

SubspaceDashboard.run(__name__)
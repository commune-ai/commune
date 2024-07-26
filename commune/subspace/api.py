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
    
    def __init__(self, root_netuid=0, max_age = 10000, backend=None):
        self.subspace = c.module('subspace')()
        self.max_age = max_age
        self.root_netuid = root_netuid
        self.global_state()
        c.thread(self.sync_loop)

    def resolve_backend(self, backend):
        if backend != None:
            backend = c.connect(backend)
        else:
            backend = self.backend
        return backend

    def global_state(self, max_age=None, update=False):
        max_age = max_age or self.max_age

        global_state = self.get('global_state', None, max_age=max_age, update=update)

        if global_state == None :
            params = self.subspace.global_params(max_age=max_age)
            subnet2netuid = self.subspace.subnet2netuid(max_age=max_age)
            subnet_names = list(subnet2netuid.keys())
            netuids = list(subnet2netuid.values())
            subnet2emission = self.subspace.subnet2emission(max_age=max_age)
            global_state =  {
                'params': params,
                'subnet2netuid': subnet2netuid,
                'subnet_names': subnet_names,
                'netuids': netuids,
                'subnet2emission': subnet2emission
            }
        self.__dict__.update(global_state)
        return global_state

    def subnet_state(self, netuid=0, max_age=None):
        max_age = max_age or self.max_age
        subnet_state = self.get(f'subnet_state/{netuid}', None, netuid=netuid, max_age=max_age)
        if subnet_state == None:
            subnet_params = self.subspace.subnet_params(netuid=netuid, max_age=max_age)
            subnet_modules = self.subspace.get_modules(netuid=netuid,  max_age=max_age)
            subnet_name = subnet_params['name']
            subnet_state = {
                'params': subnet_params,
                'netuid': netuid,
                'name': subnet_name,
                'modules': subnet_modules
            }
        return subnet_state
    
    def sync_loop(self):
        while True:
            self.sync(background=False)
            c.print('Synced all subnets, sleeping')
            c.sleep(self.max_age)



    sync_futures = []
    def sync(self, max_age=1000, timeout=60, sleep=10, background=True):
        if len(self.sync_futures) > 0:
            c.print('Waiting for previous sync to complete')
            c.wait(self.sync_futures)
        if background:
            future = c.submit(self.sync, params={'max_age': max_age, 'timeout': timeout, 'sleep': sleep})
            self.sync_futures += [future]
            return {'msg': 'syncing', 'background': True}
    
        c.sleep(5)
        self.global_state(**{'max_age': max_age})
        for netuid in self.netuids:
            c.print(f"Syncing {netuid}")
            self.subnet_state(**{'netuid': netuid, 'max_age': max_age})     
        return {'msg': 'synced', 'netuids': self.netuids, 'subnet_names': self.subnet_names}
    
    

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
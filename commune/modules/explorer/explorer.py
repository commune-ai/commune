
import scalecodec
from retry import retry
import commune as c

import streamlit as st
import pandas as pd


class Explorer(c.Module):
   
    def __init__(self, *args, **kwargs):
        self.subspace = c.module('subspace')(*args, **kwargs)

    @classmethod
    def dashboard(cls):
        self = cls()
        # st.write(self.subnets)
        subnet_df = pd.DataFrame(self.subnets)
        modules = self.modules
        self.subnet_names = [s['name'] for s in self.subnets]
        self.netuid2subnet = {i:s for i,s in enumerate(self.subnet_names)}
        self.subnet2netuid = {s:i for i,s in enumerate(self.subnet_names)}
        subnet = st.selectbox('Select Subnet', self.subnet_names, self.netuid, key='subnet.sleect')
        netuid = self.subnet2netuid[subnet]

        modules = self.state['modules'][netuid]
        for i in range(len(modules)):
            for k in [ 'key', 'address', 'stake_from']:
                modules[i].pop(k, None)
            for k in ['emission', 'stake']:
                modules[i][k] = modules[i][k]/1e9
        df = pd.DataFrame(modules)

        with st.expander('Modules', expanded=True):
            search = st.text_input('Search Namespace', '', key='search.namespace.subnet')
            if search != '':
                df = df[df['name'].str.contains(search)]
            n = st.slider('n', 1, len(df), 100, 1, key='n.modules')
            st.write(df[:n])

        
        self.subnet_info = {}
        self.subnet_info['params'] = self.state['subnets'][self.netuid]
        num_rows = 4
        num_cols = len(subnet_df.columns) // num_rows 
        
        with st.expander('Subnet Params', expanded=False):
            self.subnet_info['n'] = len(modules)
            self.subnet_info['total_stake'] = sum([m['stake'] for m in modules])
            subnet_params = self.subnet_info['params']
            cols = st.columns(num_cols)
            for i, (k,v) in enumerate(subnet_params.items()):
                with cols[i % num_cols]:
                    st.write(k)
                    st.code(v)

Explorer.run(__name__)

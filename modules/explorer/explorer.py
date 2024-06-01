
import scalecodec
from retry import retry
import commune as c

import streamlit as st
import pandas as pd


class Explorer(c.Module):
   
    def __init__(self, *args, **kwargs):
        self.subspace = c.module('subspace')(*args, **kwargs)



    def select_network(self, update:bool=False, netuid=0, network='main', state=None, _self = None):
        
        if _self != None:
            self = _self

        self.key = c.get_key()
        @st.cache_data(ttl=60*60*24, show_spinner=False)
        def get_networks():
            chains = c.chains()
            return chains
        self.networks = get_networks()
        self.network = st.selectbox('Select Network', self.networks, 0, key='network')

        @st.cache_data(show_spinner=False)
        def get_state(network):
            return c.get_state(network, netuid='all')

        state  =  get_state(self.network)
        subnets = state['subnets']
        name2subnet = {s['name']:s for s in subnets}
        name2netuid = {s['name']:i for i,s in enumerate(subnets)}
        subnet_names = list(name2subnet.keys())
        subnet_name = st.selectbox('Select Subnet', subnet_names, 0, key='subnet.sidebar')
        self.netuid = name2netuid[subnet_name]
        modules = state['modules']

        for subnet in subnets:
            s_name = subnet['name']
            s_netuid = name2netuid[s_name]
            s_modules = modules[s_netuid]
            subnet = subnets[s_netuid]
            subnet['n'] = len(s_modules)
            total_stake = sum([sum([v/1e9 for k,v in m['stake_from'].items()]) for m in s_modules])
            subnet['total_stake'] = total_stake
            subnet['emission_per_epoch'] = sum([m['emission']/1e9  for m in s_modules])
            subnet['emission_per_block'] = subnet['emission_per_epoch'] / subnet['tempo']
            subnet['emission_per_day'] = subnet['emission_per_block'] * state['blocks_per_day']
            
            subnet['n'] = len(state['modules'][self.netuid])
 


        self.subnet = subnet
        self.subnets = subnets
        self.modules = modules


    def modules_dashboard(self):
        modules = self.modules[self.netuid]
        for m in modules:
            m.pop('stake_from', None)
        df = pd.DataFrame(self.modules[self.netuid])
        cols_options = list(df.columns)
        default_cols = cols_options[:10]
        selected_cols = st.multiselect('Select Columns', cols_options, default_cols, key='select.columns.modules')
        search = st.text_input('Search', '', key='search.namespace.subnet')
        if search != '':
            # search across ['name', 'address', 'key]
            search_cols = ['name', 'address', 'key']
            search_df = df[search_cols]
            search_df = search_df[search_df['name'].str.contains(search) | search_df['address'].str.contains(search) | search_df['key'].str.contains(search)]
            df = df[df.index.isin(search_df.index)]
        df = df[selected_cols]
       
        n = st.slider('n', 1, len(df), 100, 1, key='n.modules')
        st.write(df[:n])


    def subnet_dashboard(self):

        subnet_df = pd.DataFrame(self.subnets)
        df_cols = list(subnet_df.columns)


        default_cols = ['name', 'total_stake',  'n', 'emission_per_epoch', 'founder', 'founder_share', 'tempo']
        selected_cols = st.multiselect('Select Columns', df_cols, default_cols)
        cols = st.columns(2) 

        # sort_cols = cols[0].multiselect('Sort Columns', df_cols, ['total_stake'])
        # ascending = cols[1].checkbox('Ascending', False)
        search = st.text_input('Search', '', key='search.subnet')
        if search != '':
            subnet_df = subnet_df[subnet_df['name'].str.contains(search)]
        st.write(subnet_df[selected_cols])



    @classmethod
    def dashboard(cls):
        '''
        hey
        '''
        #
        self = cls()
        # st.write(self.subnets)

        with st.sidebar:
            self.select_network()

        tab_names =  ['Subnet', 'Modules']
        tabs = st.tabs(tab_names)
        for i, tab in enumerate(tabs) :
            tab_name = tab_names[i].lower()
            with tab:
                getattr(self, f'{tab_name}_dashboard')()
        
 

Explorer.run(__name__)

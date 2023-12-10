import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px


class Dashboard(c.Module):
    
    def __init__(self, netuid = 0, network = 'main', ): 
        self.set_config(locals())
    
        st.set_page_config(layout="wide")
        self.st = c.module('streamlit')()
        self.st.load_style()

        # THIS IS A SPECIAL FUNCTION
        self.load_state(update=False)


    def sidebar(self, sidebar:bool = True):
        if sidebar:
            with st.sidebar:
                return self.sidebar(sidebar=False)
        st.title(f'COMMUNE')
        self.select_key()
        self.select_network()

    def select_network(self, network=None):
        if network == None:
            network = self.network
        if not c.server_exists('module'):
                c.serve(wait_for_server=True)

        self.networks = c.networks()
        network2index = {n:i for i,n in enumerate(self.networks)}
        index = network2index['local']
        self.network = st.selectbox('Select a Network', self.networks, index=index, key='network.sidebar')
        namespace = c.namespace(network=self.network)
        with st.expander('Namespace', expanded=True):
            search = st.text_input('Search Namespace', '', key='search.namespace')
            df = pd.DataFrame(namespace.items(), columns=['key', 'value'])
            if search != '':
                df = df[df['key'].str.contains(search)]
            st.write(f'**{len(df)}** servers with **{search}** in it')

            df.set_index('key', inplace=True)
            st.dataframe(df, width=1000)
        sync = st.button(f'Sync {self.network} Network'.upper(), key='sync.network')
        self.servers = c.servers(network=self.network)
        if sync:
            c.sync()




    def get_module_stats(self, modules):
        df = pd.DataFrame(modules)
        del_keys = ['stake_from', 'stake_to', 'key']
        for k in del_keys:
            del df[k]
        return df



    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        self = cls()
        self.sidebar()


    def select_key(self):
        keys = c.keys()
        key2index = {k:i for i,k in enumerate(keys)}
        self.key = st.selectbox('Select Key', keys, key2index['module'], key='key.sidebar')
        key_address = self.key.ss58_address
        st.write('address')
        st.code(key_address)
        return self.key

    @classmethod
    def dashboard(cls):
        self = cls()
        self.sidebar()

        tabs = st.tabs(['SERVE', 'WALLET', 'PLAYGROUND', 'REMOTE', 'CHAT'])
        chat = False
        with tabs[0]: 
            self.modules_dashboard()  
        with tabs[1]:
            self.subspace_dashboard()
        with tabs[2]:
            self.playground_dashboard()
        with tabs[3]:
            self.remote_dashboard()
        
        if chat:
            self.chat_dashboard()
 
    def remote_dashboard(self):
        st.write('# Remote')

        # return c.module('remote').dashboard()


        


    def subspace_dashboard(self):
        return c.module('subspace.dashboard').dashboard(key=self.key)
    
    def tokenomics_dashboard(self):
        return c.module('subspace.tokenomics').dashboard(key=self.key)
    
    @classmethod
    def dash(cls, *args, **kwargs):
        c.print('FAM')
        return cls.st('dashboard')
    
Dashboard.run(__name__)


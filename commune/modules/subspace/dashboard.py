import commune as c
import streamlit as st


class SubspaceDashboard(c.Module):
    def __init__(self, config=None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.keys = list(c.keys()) 
        self.set_subsapce()
        
    def set_subsapce(self, network='subspace'):
        self.subspace = c.module(network)()

        self.chain_info = {
            'block': self.subspace.block,
        }
        
        
        
    @classmethod
    def dashboard(cls):
        self = cls()

        st.header('Subspace Dashboard')
        self.subspace
        
        
        st.write(self.subspace)
        key = st.selectbox('Key', self.keys)
        st.write(key)
        
        st.write(c.keys())
        
        with st.sidebar:
            self.sidebar()
        
        
SubspaceDashboard.run(__name__)



        
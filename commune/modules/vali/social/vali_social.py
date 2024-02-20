import commune as c
import streamlit as st

class ValiSocial(c.Module):
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)

    @classmethod
    def dashboard(cls):
        self = cls()
        st.write("vali social run")
       
        subspace = c.module('subspace')()


        features = ['key',  
                'address', 
                'name', 
                'emission', 
                'incentive', 
                'dividends', 
                'last_update', 
                'delegation_fee',
                'trust', 
                'regblock']
        
        default = [ 'name', 'emission', 'incentive', 'dividends', 'last_update', 'key',  'delegation_fee', 'regblock', 'trust', ]
        features = st.multiselect("Select features", features, default)
        update = st.button("Update")
        if not c.server_exists('subspace'):
            c.serve('subspace', wait_for_server=True)
            c.submit('subspace/sync')
            return
        modules = c.submit(subspace.modules, kwargs=dict(features=features), timeout=20)
        modules = c.wait(modules)
        df = c.df(modules)
        
        st.write(df)



ValiSocial.run(__name__)
    
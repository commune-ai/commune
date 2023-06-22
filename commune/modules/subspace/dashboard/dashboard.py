import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px


class SubspaceDashboard(c.Module):
    
    def __init__(self, config=None): 
        st.set_page_config(layout="wide")


        c.module('streamlit').load_style()
        self.set_config(config=config)
        self.subspace = c.module('subspace')()

        
        
    

    def key_dashboard(self):
        keys = c.keys()
        key = None
        with st.expander('Select Key', expanded=True):

            key2index = {k:i for i,k in enumerate(keys)}
            if key == None:
                key = keys[0]
            key = st.selectbox('Select Key', keys, index=key2index[key])
                    
            key = c.get_key(key)
                
            st.write(key)
            self.key = key
            
            
        with st.expander('Create Key', expanded=False):                
            new_key = st.text_input('Name of Key', '', key='create')
            create_key_button = st.button('Create Key')
            if create_key_button and len(new_key) > 0:
                c.add_key(new_key)
                key = c.get_key(new_key)
                
        with st.expander('Remove Key', expanded=False):                
            rm_keys = st.multiselect('Select Key(s) to Remove', keys, [], key='rm_key')
            rm_key_button = st.button('Remove Key')
            if rm_key_button:
                c.rm_keys(rm_keys)
                            

    default_network = 'commune'
    def network_management(self):
        with st.expander('Network', expanded=True):
        
            networks = self.subspace.subnets()
            if len(networks) == 0:
                networks = [self.default_network]
            else:
                networks = [n['name'] for n in networks]
            network2index = {n:i for i,n in enumerate(networks)}
            network = st.selectbox('Network', networks, index=network2index['commune'])
            self.netuid = self.subspace.subnet2netuid(network)
            
           
        
    def sidebar(self):
        with st.sidebar:
            st.write('#### Subspace')
            self.key_management()
            self.network_management()
                
            update = st.button('Update')
            if update:
                self.subspace = c.module('subspace')()
                self.registered_keys = self.subspace.registered_keys()
                st.write(self.registered_keys)
          

    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        self = cls()
        # self.sidebar()
        
        tabs = st.tabs(['Modules', 'Network', 'Keys']) 
        with tabs[0]:   
            st.write('# Modules')
            self.module_dashboard()
        with tabs[1]:
            self.network_dashboard()
        with tabs[2]:
            self.key_dashboard()
            
        # with st.expander('Transfer Module', expanded=True):
        #     self.transfer_dashboard()
        # with st.expander('Staking', expanded=True):
        #     self.staking_dashboard()
        

    def network_dashboard(self):
        st.write('# Network')
        
        self.subnets = self.subspace.subnets()
        df = pd.DataFrame(self.subnets)
        st.write(df)
        if len(df) > 0:
            fig = px.pie(df, values='stake', names='name', title='Subnet Balances')
            st.plotly_chart(fig)
        
        
        for subnet in self.subnets:
            network = subnet.pop('name', None)
            with st.expander(network, expanded=True):
                st.write(subnet)
        
        # convert into metrics
        
        

    def transfer_dashboard(self):
            kwargs = self.function2streamlit(module='subspace', fn='transfer', skip_keys = ['key', 'wait_for_finalization', 'prompt', 'keep_alive', 'wait_for_inclusion'])
            if not c.is_number(kwargs['amount']):
                st.error('Amount must be a number')
            else:
                kwargs['amount'] = float(kwargs['amount'])  
            transfer_button = st.button('Transfer')
            if transfer_button:
                self.transfer(**kwargs)
            
            

    def staking_dashboard(self):
        st.write('#### Staking')
        stake_kwargs = self.function2streamlit(module='subspace', fn='add_stake', skip_keys = ['key', 'wait_for_finalization', 'prompt', 'keep_alive', 'wait_for_inclusion'])
        
        stake_kwargs['netuid'] = self.netuid
        st.write('#### Unstaking')
        unstake_kwargs = self.function2streamlit(module='subspace', fn='unstake', skip_keys = ['key', 'wait_for_finalization', 'prompt', 'keep_alive', 'wait_for_inclusion'])


    def subnet_dashboard(self):
        st.write(self.subspace.subnets())

    def module_dashboard(self):
        
        self.modules = list(self.subspace.modules(fmt='token').values())
        self.namespace = self.subspace.namespace()
        df = pd.DataFrame(self.modules)
        
        with st.expander('Register Module', expanded=True):
            self.launch_dashboard()
            
        with st.expander('Modules', expanded=True):
            st.write(self.modules)
            
        # pie of stake per module
        # select fields to show
        
        if len(df)> 0:
            with st.expander('Module Statistics', expanded=True):
                value_field2index = {v:i for i,v in enumerate(list(df.columns))}
                key_field2index = {v:i for i,v in enumerate(list(df.columns))}
                value_field = st.selectbox('Value Field', df.columns , index=value_field2index['stake'])
                key_field = st.selectbox('Key Field',df.columns, index=value_field2index['name'])
                
                # plot pie chart in a funky color

                fig = px.pie(df, values=value_field, names=key_field, title='Module Balances', color_discrete_sequence=px.colors.sequential.RdBu)
                # show the key field
                # do it in funky way
                
                st.plotly_chart(fig)
                st.write(df)
        
    
    def launch_dashboard(self):
        modules = c.leaves()
        module2idx = {m:i for i,m in enumerate(modules)}


        cols = st.columns(2)
        module  = name = cols[0].selectbox('Select A Module', modules, module2idx['model.openai'])

        
        network = st.text_input('network', self.config.default_network, key='network')
        c_st = c.module('streamlit')
        c_st.line_seperator()
        st.write(f'#### Module ({module}) Kwargs ')
        kwargs = c_st.function2streamlit(module=module, fn='__init__' )
        c_st.line_seperator()


        register = st.button('Register')
        tag = cols[1].text_input('tag', 'None', key='tag')
        if 'None' == tag:
            tag = None
            
            

        if register:
            try:
                self.subspace.register(module=module, name=name, tag=tag, network=network, kwargs=kwargs)
            except Exception as e:
                st.error(e)

SubspaceDashboard.run(__name__)



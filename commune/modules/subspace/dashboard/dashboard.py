import commune as c
import streamlit as st
from streamlit.components.v1 import components


class SubspaceDashboard(c.Module):
    
    def __init__(self, config=None): 
        st.set_page_config(layout="wide")


        c.module('streamlit').load_style()
        self.set_config(config=config)
        self.subspace = c.module('subspace')()
        st.write(self.subspace.registered_keys())

        
        
    
        
    def line_seperator(cls, text='-', length=50):
        st.write(text*length)
    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__', 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                            mode = 'pm2'):
        
        key_prefix = f'{module}.{fn}'
        if module == None:
            module = cls
            
        elif isinstance(module, str):
            module = c.module(module)
        
        config = module.config(to_munch=False)
        
        fn_schema = module.schema(include_default=True)[fn]

        if fn == '__init__':
            extra_defaults = config
        elif extra_defaults is None:
            extra_defaults = {}

        kwargs = {}
        fn_schema['default'].pop('self', None)
        fn_schema['default'].pop('cls', None)
        fn_schema['default'].update(extra_defaults)
        fn_schema['default'].pop('config', None)
        fn_schema['default'].pop('kwargs', None)
        
        
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        if cols == None:
            cols = [1 for i in list(range(int(len(fn_schema['input'])**0.5)))]
        cols = st.columns(cols)


        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
            if fn_key in skip_keys:
                continue
            if k in fn_schema['input']:
                k_type = fn_schema['input'][k]
                if 'Munch' in k_type or 'Dict' in k_type:
                    k_type = 'Dict'
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            

            
            col_idx = col_idx % (len(cols))
            kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}')
            
            
        return kwargs
    
    def key_management(self):
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
            rm_key = st.selectbox('Select Key to Remove', keys, index=0)
            rm_key_button = st.button('Remove Key')
            if rm_key_button:
                c.rm_key(rm_key)
                            

    default_network = 'commune'
    def network_management(self):
        with st.expander('Network', expanded=True):
        
            networks = self.subspace.subnets()
            if len(networks) == 0:
                networks = [default_network]
            else:
                networks = [n['name'] for n in networks]
            network2index = {n:i for i,n in enumerate(networks)}
            network = st.selectbox('Network', networks, index=network2index['commune'])
            
           
        
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
        import plotly.express as px
        self = cls()
        self.sidebar()
        with st.expander('Register Module', expanded=True):
            self.register_dashboard()
        with st.expander('Transfer Module', expanded=True):
            self.transfer_dashboard()
        with st.expander('Staking', expanded=True):
            self.staking_dashboard()

    def transfer_dashboard(self):
            kwargs = self.function2streamlit(module='subspace', fn='transfer', skip_keys = ['key', 'wait_for_finalization', 'prompt', 'keep_alive', 'wait_for_inclusion'])
            if not c.is_number(kwargs['amount']):
                st.error('Amount must be a number')
            else:
                kwargs['amount'] = float(kwargs['amount'])  
            transfer_button = st.button('Transfer')
            if transfer_button:
                self.transfer(**kwargs)
            
            st.write(kwargs)
            

    def staking_dashboard(self):
        st.write('#### Staking')
        stake_kwargs = self.function2streamlit(module='subspace', fn='add_stake', skip_keys = ['key', 'wait_for_finalization', 'prompt', 'keep_alive', 'wait_for_inclusion'])
        st.write('#### Unstaking')
        unstake_kwargs = self.function2streamlit(module='subspace', fn='unstake', skip_keys = ['key', 'wait_for_finalization', 'prompt', 'keep_alive', 'wait_for_inclusion'])


    def register_dashboard(self, key=None):
        st.write('# Register Module')
        modules = c.leaves()
        module2idx = {m:i for i,m in enumerate(modules)}
        cols = st.columns(3)
        module  = name = cols[0].selectbox('module', modules, module2idx['model.openai'])
        network = cols[1].text_input('network', self.config.default_network, key='network')
        tag = cols[2].text_input('tag', 'None', key='tag')
        serve = st.button('Register')

        self.line_seperator()
        st.write('#### Module Arguments')
        kwargs = self.function2streamlit(module=module, fn='__init__' )
        self.line_seperator()
        if 'None' == tag:
            tag = None
            
        
        if serve:
            self.subspace.register(module=module, name=name, tag=tag, key=self.key, network=network, kwargs=kwargs)
        
SubspaceDashboard.run(__name__)



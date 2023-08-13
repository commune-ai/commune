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
        self.sync()


        
    def sync(self,):
        self.subspace = c.module('subspace')()
        self.modules = self.subspace.modules(fmt='token')
        self.state = self.subspace.state_dict()
        
        self.netuid = self.config.netuid
        self.subnets = self.state['subnets']
        self.subnet = self.subnets[self.netuid]
        self.namespace = {m['name']: m['address'] for m in self.modules}
        self.modules = self.state['modules'][self.netuid]
        
           
    

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
                            

    default_subnet = 'commune'
    def subnet_management(self):
        with st.expander('Subnet', expanded=True):
        
            subnets = self.subspace.subnets()
            if len(subnets) == 0:
                subnets = [self.default_subnet]
            else:
                subnets = [n['name'] for n in subnets]
            subnet2index = {n:i for i,n in enumerate(subnets)}
            subnet = st.selectbox('Subnet', subnets, index=subnet2index['commune'])
            self.netuid = self.subspace.subnet2netuid(subnet)
            
           
           
    
        
    def sidebar(self):
        with st.sidebar:
            update = st.button('Update')

    def my_modules_dashboard(self):
        st.write('#### My Modules')

        modules = [m for m in self.subspace.my_modules(names_only=False)]
        for module in modules:
            with st.expander(module['name'], expanded=False):
                st.write(module)
            

    def validator_dashboard(self):
        pass

    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        
        self = cls()
        st.write('# Subspace')

        self.sidebar()
        
        tabs = st.tabs(['Modules', 'Validators', 'Playground']) 
        with tabs[0]:   
            st.write('# Modules')
            self.modules_dashboard()
        with tabs[1]:
            self.validator_dashboard()
        with tabs[2]:
            self.playground_dashboard()
            
        # with st.expander('Transfer Module', expanded=True):
        #     self.transfer_dashboard()
        # with st.expander('Staking', expanded=True):
        #     self.staking_dashboard()
        

    def subnet_dashboard(self):
        st.write('# Subnet')
        
        df = pd.DataFrame(self.subnets)
        st.write(df)
        if len(df) > 0:
            fig = px.pie(df, values='stake', names='name', title='Subnet Balances')
            st.plotly_chart(fig)
        
        
        for subnet in self.subnets:
            subnet = subnet.pop('name', None)
            with st.expander(subnet, expanded=True):
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
                self.subspace.transfer(**kwargs)
            
    @classmethod
    def process_kwargs(cls, kwargs:dict, fn_schema:dict):
        
        for k,v in kwargs.items():
            if v == 'None':
                v = None
                
            elif k in fn_schema['input'] and fn_schema['input'][k] == 'str':
                if v.startswith("f'") or v.startswith('f"'):
                    v = c.ljson(v)
                elif v.startswith('[') and v.endswith(']'):
                    v = v
                elif v.startswith('{') and v.endswith('}'):
                    v = v
                else:
                    v = v
                
            elif k == 'kwargs':
                continue
            elif v == 'NA':
                assert k != 'NA', f'Key {k} not in default'
            elif v in ['True', 'False']:
                v = eval(v)
            else:
                try:
                    v = eval(v) 
                except:
                    pass
            
            kwargs[k] = v
        return kwargs
    
    def playground_dashboard(self):
        st.write('# Playground')

    def register_dashboard(self):
        
        df = pd.DataFrame(self.modules)
        
        with st.expander('Register Module', expanded=True):
            self.launch_dashboard()
            
        with st.expander('Modules', expanded=False):
            st.write(self.modules)
            
        # pie of stake per module
        # select fields to show
        
        if len(df)> 0:
            with st.expander('Module Statistics', expanded=False):
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
        
    def modules_dashboard(self):
        modules = c.modules()
        module2idx = {m:i for i,m in enumerate(modules)}


        cols = st.columns(2)
        module  = name = cols[0].selectbox('Select A Module', modules, module2idx['model.openai'])

        
        subnet = st.text_input('subnet', self.config.subnet, key='subnet')
        c_st = c.module('streamlit')
        c_st.line_seperator()
        st.write(f'#### Module ({module}) Kwargs ')
        
        fn_schema = c.get_function_schema(c.module(module), '__init__')
        kwargs = c_st.function2streamlit(module=module, fn='__init__' )
        c_st.line_seperator()

        st.write(self.process_kwargs(kwargs , fn_schema))
        register = st.button('Register')
        tag = cols[1].text_input('tag', 'None', key='tag')
        if 'None' == tag:
            tag = No
            
        if 'tag' in kwargs:
            kwargs['tag'] = tag

        if register:
            st.write(self.subspace)
            response = self.subspace.register(module=module,  tag=tag, subnet=subnet, kwargs=kwargs, network=self.config.network)
           
            if response['success']:
                st.success('Module Registered')
            else:
                st.error(response['message'])
                




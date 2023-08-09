import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px


class SubspaceDashboard(c.Module):
    
    def __init__(self, config=None): 
        st.set_page_config(layout="wide")
        self.st = c.module('streamlit')()
        self.st.load_style()
        self.set_config(config=config)
        self.load_state(sync=False)
        self.key = c.get_key()


    def sync(self):
        return self.load_state(sync=True)
    
    def load_state(self, sync=True):
        self.subspace = c.module('subspace')()
        if sync:
            self.subspace.sync()
        self.state = self.subspace.state_dict()
        self.netuid = self.config.netuid
        self.subnets = self.state['subnets']

        for i, s in enumerate(self.subnets):
            for k in ['stake', 'emission', 'ratio']:
                self.subnets[i][k] = self.subspace.format_amount(s[k], fmt='j')
            
        self.subnet2info = {s['netuid']: s for s in self.subnets}
        self.subnet2netuid = {s['name']: s['netuid'] for s in self.subnets}
        self.subnet_names = [s['name'] for s in self.subnets]
        self.my_keys = self.subspace.my_keys()
        self.my_modules = self.subspace.my_modules(fmt='j')
    @property
    def module_names(self):
        return [m['name'] for m in self.modules]

    @property
    def subnet(self):
        if not hasattr(self, '_subnet'):
            self._subnet = self.subnet_names[0]
        return self.state['subnets'][self.netuid]['name']
    
    @subnet.setter
    def subnet(self, subnet):
        self.netuid = self.subnet2netuid[subnet]
        self._subnet = subnet
        return self._subnet
    




    @property
    def namespace(self):
        return {m['name']: m['address'] for m in self.modules}

    @property
    def modules(self):
        return self.state['modules'][self.netuid]
        
    @property    
    def subnet_info(self):
        subnet_info =  self.subnet2info[self.netuid]
        return subnet_info
    

    def key_dashboard(self):
        keys = c.keys()
        key = None
        with st.expander('Select Key', expanded=True):

            key2index = {k:i for i,k in enumerate(keys)}
            if key == None:
                key = keys[0]
            key = st.selectbox('Select Key', keys, index=key2index[key])
                    
            key = c.get_key(key)

                
            self.key = key

            
            

            cols = st.columns(2)
            self.key_info = {
                'stake': c.round_decimals(self.subspace.get_stake(key),2),
                'balance': c.round_decimals(self.subspace.get_balance(key), 2)
            }
            cols[0].metric('Stake', self.key_info['stake'])
            cols[1].metric('Balance', self.key_info['balance'])
            
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
            keys = c.keys()
            key2index = {k:i for i,k in enumerate(keys)}
            self.subnet = st.selectbox('Select Subnet', self.subnet_names, 0)
            self.netuid = self.subnet2netuid[self.subnet]
            self.key_dashboard()
            sync = st.button('Sync')
            if sync:
                self.sync()

    def stats_dashboard(self, max_rows=100):
        my_modules = self.modules
        df = c.df(my_modules)
        for k in ['key', 'balance', 'address']:
            del df[k]
    
        for k in ['emission', 'stake']:
            df[k] = df[k].apply(lambda x: c.round_decimals(self.subspace.format_amount(x, fmt='j'), 2))

        df.sort_values('incentive', inplace=True, ascending=False)
        df = df[:max_rows]
        st.dataframe(df, width=1000)



        # BAR OF INCENTIVES
        options = ['emission', 'incentive', 'dividends']
        selected_keys = st.multiselect('Select Columns', options, options, key='stats')

        for i, k in enumerate(selected_keys):
            cols = st.columns(2)

            fig = px.line(df, y=k, x= 'name', title=f'{k.upper()} Per Module')
            cols[0].plotly_chart(fig)
            fig = px.histogram(df, y=k, title=f'{k.upper()} Distribution')
            cols[1].plotly_chart(fig)





    def validator_dashboard(self):
        pass

    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        
        self = cls()

        self.sidebar()
        
        tabs = st.tabs(['Modules', 'Validators', 'Stats', 'Playground']) 
        with tabs[0]:   
            self.modules_dashboard()
        with tabs[1]:
            self.validator_dashboard()
        with tabs[2]:
            self.stats_dashboard()
        with tabs[3]:
            self.playground_dashboard()

            
        # with st.expander('Transfer Module', expanded=True):
        #     self.transfer_dashboard()
        # with st.expander('Staking', expanded=True):
        #     self.staking_dashboard()
        

    def subnet_dashboard(self):
        st.write('# Subnet')
        
        df = pd.DataFrame(self.subnets)
        if len(df) > 0:
            fig = px.pie(df, values='stake', names='name', title='Subnet Balances')
            st.plotly_chart(fig)
        
        
        for subnet in self.subnets:
            subnet = subnet.pop('name', None)
            with st.expander(subnet, expanded=True):
                st.write(subnet)
        
        # convert into metrics
        
    def transfer_dashboard(self):
        with st.expander('Transfer', expanded=True):
            amount = st.number_input('amount', 0.0, 10000000.0, 0.0, 0.1)
            to_address = st.text_input('dest', '')
            transfer_button = st.button('Transfer')
            if transfer_button:
                kwargs = {
                    'dest': to_address,
                    'amount': amount,
                    'key': self.key,
                }
                self.subspace.transfer(**kwargs)

    def staking_dashboard(self):
        with st.expander('Stake', expanded=True):

            amount = st.number_input('STAKE Amount', 0.0, self.key_info['stake'], 0.0, 0.1)
            default_module = self.subspace.key2module(self.key.path)['name']
            module2index = {m:i for i,m in enumerate(self.module_names)}
            
            module = st.selectbox('Module', self.module_names, index=module2index[default_module])
            transfer_button = st.button('STAKE')

            if transfer_button:
                kwargs = {
                    'amount': amount,
                    'module_key': module,
                    'key': self.key,
                }
                st.write(kwargs)

                self.subspace.stake(**kwargs)

        with st.expander('UnStake', expanded=True):

            amount = st.number_input('UNSTAKE Amount', 0.0, self.key_info['stake'], 0.0, 1.0)
            module2index = {m:i for i,m in enumerate(self.module_names)}
            default_module = self.subspace.key2module(self.key.path)['name']

            module = st.selectbox('Module', self.module_names, index=module2index[default_module], key='unstake')
            unstake_button = st.button('UNSTAKE')
            if unstake_button:
                kwargs = {
                    'amount': amount,
                    'module_key': module,
                    'key': self.key,
                }
                self.subspace.unstake(**kwargs)


            
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

        self.launch_dashboard(expanded=False)

        if self.subspace.is_registered(self.key):
            self.staking_dashboard()
            self.transfer_dashboard()
            
    def launch_dashboard(self, expanded=True):
        modules = c.modules()


        with st.expander('Launch', expanded=expanded):
            module2idx = {m:i for i,m in enumerate(modules)}

            

            
            st.write(f'#### Miner Launcher ')
            self.st.line_seperator()

            cols = st.columns([3,1, 6])
            


            with cols[0]:
                module  = st.selectbox('Select A Module', modules, module2idx['model.openai'])
                tag = self.subspace.get_unique_tag(module=module)
                subnet = st.text_input('subnet', self.config.subnet, key='subnet')
                tag = st.text_input('tag', tag, key='tag')
                n = st.slider('replicas', 1, 10, 1, 1)
                serve = st.checkbox('serve', True)
    
                register = st.button('Register')

        
            with cols[-1]:
                st.write(f'#### {module.upper()} Kwargs ')

                fn_schema = c.get_function_schema(c.module(module), '__init__')
                kwargs = self.st.function2streamlit(module=module, fn='__init__' )

                kwargs = self.st.process_kwargs(kwargs, fn_schema)
                self.st.line_seperator()


            
            if 'None' == tag:
                tag = None
                
                
            if 'tag' in kwargs:
                kwargs['tag'] = tag

            if register:
                try:
                    if n > 1:
                        tags = [f'{tag}.{i}' if tag != None else str(i) for i in range(n)]
                    else:
                        tags = [tag]

                    for tag in tags:
                        response = self.subspace.register(module=module, 
                                                           tag=tag, 
                                                          subnet=subnet, 
                                                          kwargs=kwargs, 
                                                          network=self.config.network, 
                                                          serve=serve)
                except Exception as e:
                    response = {'success': False, 'message': str(e)}
                if response['success']:
                    st.success('Module Registered')
                else:
                    st.error(response['message'])
                    




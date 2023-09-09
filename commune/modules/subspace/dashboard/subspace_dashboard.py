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
        self.st = c.module('streamlit')()


    def sync(self):
        return self.load_state(sync=True)
    
    


    def load_state(self, sync:bool=False):


        self.subspace = c.module('subspace')()

        

        self.state = self.subspace.state_dict()

        
        self.netuid = self.config.netuid
        self.subnets = self.state['subnets']

        self.subnet2info = {s['netuid']: s for s in self.subnets}
        self.subnet2netuid = {s['name']: s['netuid'] for s in self.subnets}
        self.subnet_names = [s['name'] for s in self.subnets]
        self.my_keys = self.subspace.my_keys()
        self.modules = self.state['modules'][self.netuid]
        self.validators = [m for m in self.modules if m['name'].startswith('vali') ]
        self.keys  = c.keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}

        self.subnet = self.subnet_names[0]
        self.namespace = {m['name']: m['address'] for m in self.modules}
        self.module_names = [m['name'] for m in self.modules]
        self.subnet_info =  self.subnet2info[self.netuid]
    

    def select_key(self,):
        with st.expander('Select Key', expanded=True):
            key = 'module'
            key = st.selectbox('Select Key', self.keys, index=self.key2index[key])
            self.key =  c.get_key(key)
            if self.key.path == None:
                self.key.path = key
            self.key_info_dict = self.subspace.key_info(self.key.path, fmt='j')

            st.write('Address: ', self.key.ss58_address)
            st.write('Stake', self.key_info_dict.get('stake', 0))
            st.write('Balance', self.key_info_dict.get('balance', 0))

    def create_key(self):
        with st.expander('Create Key', expanded=False):                
            new_key = st.text_input('Name of Key', '', key='create')
            create_key_button = st.button('Create Key')
            if create_key_button and len(new_key) > 0:
                c.add_key(new_key)
                key = c.get_key(new_key)

    def rename_key(self):
        with st.expander('Rename Key', expanded=False):    
            old_key = st.selectbox('Select Key', self.keys, index=self.key2index[self.key.path], key='select old rename key')           
            new_key = st.text_input('New of Key', '', key='rename')
            rename_key_button = st.button('Rename Key')
            replace_key = st.checkbox('Replace Key')
            if rename_key_button and len(new_key) > 0:
                if c.key_exists(new_key) and not replace_key:
                    st.error('Key already exists')
                c.rename_key(old_key,new_key)
                key = c.get_key(new_key)
    
    def remove_key(self):       
        with st.form(key='Remove Key'):            
            rm_keys = st.multiselect('Select Key(s) to Remove', self.keys, [], key='rm_key')
            rm_key_button = st.form_submit_button('Remove Key')
            if rm_key_button:
                c.rm_keys(rm_keys)

    def key_dashboard(self):
        # self.select_key()
        self.create_key()
        self.rename_key()
        self.remove_key()

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
            st.write('# commune')
            key2index = {k:i for i,k in enumerate(self.keys)}
            st.write('## Select Subnet')
            self.subnet = st.selectbox(' ', self.subnet_names, 0, key='Select Subnet')
            self.netuid = self.subnet2netuid[self.subnet]
            sync = st.button('Sync')
            if sync:
                self.sync()
        
            self.select_key()

    def get_module_stats(self, modules):
        df = pd.DataFrame(modules)
        del_keys = ['stake_from', 'stake_to', 'key']
        for k in del_keys:
            del df[k]
        return df

    # for k in ['emission', 'stake']:
    #     df[k] = df[k].apply(lambda x: c.round_decimals(self.subspace.format_amount(x, fmt='j'), 2))

    # df.sort_values('incentive', inplace=True, ascending=False)
    # df = df[:max_rows]
    # st.write(df)
    # st.dataframe(df, width=1000)
    # # BAR OF INCENTIVES
    # options = ['emission', 'incentive', 'dividends']
    # selected_keys = st.multiselect('Select Columns', options, options, key='stats')

    # for i, k in enumerate(selected_keys):
    #     cols = st.columns(2)

    #     fig = px.line(df, y=k, x= 'name', title=f'{k.upper()} Per Module')
    #     cols[0].plotly_chart(fig)
    #     fig = px.histogram(df, y=k, title=f'{k.upper()} Distribution')
    #     cols[1].plotly_chart(fig)



    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        self = cls()
        st.write('Starting Dashboard')
        self.sidebar()
        
        tabs = st.tabs(['Modules', 'Validators', 'Wallet']) 
        with tabs[0]:   
            self.modules_dashboard()
        with tabs[1]:   
            self.validator_dashboard()
        with tabs[2]:
            self.wallet_dashboard()
        # with tabs[4]:
        #     self.key_dashboard()

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
        cols = st.columns(2)
        with cols[0].expander('Stake', expanded=True):

            amount = st.number_input('STAKE Amount', 0.0, float(self.key_info_dict['balance']), float(self.key_info_dict['balance']), 0.1)            
            modules = st.multiselect('Module', self.module_names, [])
            transfer_button = st.button('STAKE')

            if transfer_button:


                for m in modules:
                    kwargs = {
                        'amount': amount,
                        'module_key': m,
                        'key': self.key,
                    }

                    self.subspace.stake(**kwargs)

        with cols[1].expander('UnStake', expanded=True):
            module2stake_from_key = self.subspace.get_staked_modules(self.key, fmt='j')
            modules = list(module2stake_from_key.keys())
            module = st.selectbox('Module', modules, index=0, key='unstake')
            module_to_stake = module2stake_from_key[module]
            amount = st.number_input('UNSTAKE Amount', 0.0, float(module_to_stake), float(module_to_stake), 1.0)

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
        # self.launch_dashboard(expanded=False)
        df = self.get_module_stats(self.modules)

        archive_history = self.subspace.archive_history()
        df = c.df( archive_history)
        st.write(df)
        self.st.run(df)

        # options = ['emission', 'incentive', 'dividends', 'stake']
        # y = st.selectbox('Select Columns', options, 0)
        # # filter by stake > 1000

        # df = df[df['stake'] > 10**9]
        # histogram = px.histogram(df, x=y, title='My Modules')

        # st.write(histogram)
        
    
    def wallet_dashboard(self):
        st.write('# Wallet')
        # if self.subspace.is_registered(self.key):
        #     self.staking_dashboard()
        #     self.transfer_dashboard()
        # else:
        #     # with emoji
        #     st.error('Please Register Your Key')
    
    def validator_dashboard(self):
        validators = [{k:v[k] for k in c.copy(list(v.keys())) if k != 'stake_from'} for v in self.validators if v['stake'] > 0]
        # df = c.df(validators)
        # if len(df) == 0:
        #     st.error('No Validators')
        #     return

        # df['stake'] = df['stake']/1e9
        # df['emission'] = df['emission']/1e9
        # st.dataframe(df)
        # with st.expander('Register Validator', expanded=False):
        #     self.launch_dashboard(expanded=False, prefix='vali')
        
            
    def launch_dashboard(self, expanded=True, prefix= None ):
        modules = c.modules(prefix)
        module2idx = {m:i for i,m in enumerate(modules)}

        self.st.line_seperator()

        cols = st.columns([3,1, 6])
        


        with cols[0]:
            module  = st.selectbox('Select A Module', modules, 0)
            tag = self.subspace.resolve_unique_server_name(module, tag=None ).replace(f'{module}::', '')
            subnet = st.text_input('subnet', self.config.subnet, key=f'subnet.{prefix}')
            tag = st.text_input('tag', tag, key=f'tag.{prefix}')
            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            serve = st.checkbox('serve', True, key=f'serve.{prefix}')

            register = st.button('Register', key=f'register.{prefix}')

    
        with cols[-1]:
            st.write(f'#### {module.upper()} Kwargs ')

            fn_schema = c.fn_schema(c.module(module), '__init__')
            kwargs = self.st.function2streamlit(module=module, fn='__init__' )

            kwargs = self.st.process_kwargs(kwargs, fn_schema)
            self.st.line_seperator()

        n = 1
        
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
                    response = self.register(module=module, 
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
                




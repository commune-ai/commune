import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px


class Dashboard(c.Module):
    
    def __init__(self, netuid = 0, network = 'main', ): 
        st.set_page_config(layout="wide")
        self.set_config(locals())
        self.st = c.module('streamlit')()
        self.st.load_style()
        self.load_state(update=False)

    def sync(self):
        return self.load_state(update=True)
    
    def load_state(self, update:bool=False, netuid=0, network='main'):
        self.key = c.get_key()

        t = c.timer()

        self.subspace = c.module('subspace')()
        self.state = self.subspace.state_dict(update=update)
        c.print(f'Loaded State in {t.seconds} seconds')
        self.netuid = 0
        self.subnets = self.state['subnets']
        self.subnet = 'commune'

        self.subnet2info = {s['netuid']: s for s in self.subnets}
        self.subnet2netuid = {s['name']: s['netuid'] for s in self.subnets}
        self.subnet_names = [s['name'] for s in self.subnets]


        self.modules = self.state['modules'][self.netuid]
        self.name2key = {k['name']: k['key'] for k in self.modules}
        self.key2name = {k['key']: k['name'] for k in self.modules}

        self.keys  = c.keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}

        self.namespace = {m['name']: m['address'] for m in self.modules}
        self.module_names = [m['name'] for m in self.modules]
        self.block = self.state['block']
        for i, m in enumerate(self.modules):
            self.modules[i]['stake'] = self.modules[i]['stake']/1e9
            self.modules[i]['emission'] = self.modules[i]['emission']/1e9


        self.key_info = {
            'ss58_address': self.key.ss58_address,
            'balance': self.state['balances'].get(self.key.ss58_address,0),
            'stake_to': self.state['stake_to'][self.netuid].get(self.key.ss58_address,{}),
            
        }

        self.key_info['balance']  = self.key_info['balance']/1e9
        self.key_info['stake_to'] = {k:v/1e9 for k,v in self.key_info['stake_to']}
        # convert keys to names 
        for k in ['stake_to']:
            self.key_info[k] = {self.key2name.get(k, k): v for k,v in self.key_info[k].items()}

        total_balance = sum(self.state['balances'].values())
        self.subnet_info = self.state['subnets'][0]
        balances = self.state['balances']
        self.total_balance = sum(balances.values())/1e9
        for k in ['stake', 'emission', 'min_stake']:
            self.subnet_info[k] = self.subnet_info[k]/1e9
    def select_key(self,):
        with st.expander('Select Key', expanded=True):
            key = 'module'
            key = st.selectbox('Select Key', self.keys, index=self.key2index[key])
            self.key =  c.get_key(key)
            if self.key.path == None:
                self.key.path = key
            self.key_info_dict = {
                'balance': self.stats
            }

            st.write('Address: ', self.key.ss58_address)
            stake = sum([v for v in self.key_info.get('stake_to', {}).values()])
            st.write('Stake', stake )
            st.write('Balance', self.key_info.get('balance', 0))

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
            

    def select_network(self):            
        with st.expander('Network', expanded=True):
            st.write('# Network')
            key2index = {k:i for i,k in enumerate(self.keys)}
            self.subnet = st.selectbox(' ', self.subnet_names, 0, key='Select Subnet')
            self.netuid = self.subnet2netuid[self.subnet]
            sync = st.button('Sync')
            if sync:
                self.sync()

    def sidebar(self):
        with st.sidebar:
            self.select_key()
            self.select_network()
            st.write(self.subnet_info)

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
        self.sidebar()
        
        tabs = st.tabs(['Wallet', 'App', 'Key']) 
        with tabs[0]:
            self.wallet_dashboard()
        with tabs[1]:   
            self.modules_dashboard()
        with tabs[2]:   
            self.validator_dashboard()
        with tabs[3]:
            self.key_dashboard()

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




    def stake_dashboard(self):
        cols = st.columns(2)
        with st.expander('Stake', expanded=True):

            with st.form(key='stake'):

                amounts = st.slider('Stake Amount', 0.0,  float(self.key_info['balance']), 0.1)            
                modules = st.multiselect('Module', self.module_names, [])
                transfer_button = st.form_submit_button('STAKE')

                if transfer_button:
                    kwargs = {
                        'amounts': amounts,
                        'modules': modules,
                        'key': self.key,
                    }

                    response = self.subspace.multistake(**kwargs)
                    st.status(response)


    def unstake_dashboard(self):

        with st.expander('UnStake', expanded=True):
            module2stake_from_key = self.subspace.get_staked_modules(self.key, fmt='j')
            modules = list(self.key_info['stake_to'].keys())
            amount = st.number_input('Unstake Amount',0.0)
            modules = st.multiselect('Module', modules, [])

            unstake_button = st.button('UNSTAKE')
            if unstake_button:
                kwargs = {
                    'amounts': amount,
                    'modules': modules,
                    'key': self.key,
                }
                st.write(kwargs)
                self.subspace.multiunstake(**kwargs)


            
    
    def playground_dashboard(self):
        st.write('# Playground')


    def archive_dashboard(self):
        # self.register_dashboard(expanded=False)
        netuid = 0 
        df = self.get_module_stats(self.modules)
        archive_history = self.subspace.archive_history(lookback_hours=24, n=100, update=True)
        df = c.df(archive_history[1:])
        df['block'] = df['block'].astype(int)


        df['dt'] = pd.to_datetime(df['dt'])
        df.sort_values('block', inplace=True)
        df.reset_index(inplace=True)
        st.write(df)
        # df= df[df['market_cap'] < 1e9]


        fig = px.line(df, x='block', y='market_cap', title='Archive History')

        block2path= {b:df['path'][i] for i,b in enumerate(df['block'])}
        blocks = list(block2path.keys())
        paths = list(block2path.values())
        block = st.selectbox('Block', blocks, index=0)
        path = block2path[block]
        state = c.get(path)
        modules = state['modules'][netuid]
        for i in range(len(modules)):
            for k in ['stake_to', 'stake_from', 'key', 'address']:
                del modules[i][k]
            for k in ['emission', 'stake', 'balance']:
                modules[i][k] = modules[i][k]/1e9
        df = pd.DataFrame(modules)

        st.write(df)
        subnet_df = pd.DataFrame(state['subnets'])
        st.write(subnet_df)
        # st.write(state)

        st.write(fig)
        # options = ['emission', 'incentive', 'dividends', 'stake']
        # y = st.selectbox('Select Columns', options, 0)
        # # filter by stake > 1000

        # df = df[df['stake'] > 10**9]
        # histogram = px.histogram(df, x=y, title='My Modules')

        # st.write(histogram)
       
    def wallet_dashboard(self):




        # pie map of stake

        # remove the 
        st.write('# Wallet')
        self.register_dashboard()
        self.stake_dashboard()
        self.unstake_dashboard()
        self.transfer_dashboard()
        # else:
        #     # with emoji
        #     st.error('Please Register Your Key')

        fig = px.pie(values=list(self.key_info['stake_to'].values()), names=list(self.key_info['stake_to'].keys()), title='Stake To')
        st.plotly_chart(fig)

    
    @classmethod
    def module2streamlit(cls, module):
        if isinstance(module, str):
            module = c.module(module)
        for fn in cls.fns():
            if cls.classify_method(fn) == 'function':
                with st.expander(fn, expanded=False):
                    kwargs = cls.function2streamlit(fn)
            with st.expander(fn, expanded=False):
                kwargs = cls.function2streamlit(fn)

    def validator_dashboard(self):
        pass
    def register_dashboard(self, expanded=True, prefix= None, form = True ):


        if expanded : 
            with st.expander('Register Module', expanded=True):
                return self.register_dashboard(prefix=prefix, expanded=False)
        modules = c.modules(prefix)
        self.st.line_seperator()
        cols = st.columns([2,2,2])

        with st.form(key='register'):
            module  = cols[0].selectbox('Select A Module', modules, 0)
            tag = cols[1].text_input('tag', c.random_word(n=2), key=f'tag.{prefix}')
            stake = cols[2].number_input('stake', 0.0, 10000000.0, 0.1, key=f'stake.{prefix}')
            n = st.slider('Number of Replicas', 1, 30, 1, 1, key=f'n.{prefix}')
            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            register = st.form_submit_button('Register')

        
            st.write(f'#### {module.upper()} Kwargs ')

            fn_schema = c.fn_schema(c.module(module), '__init__')
            fns = list(fn_schema.keys())
            fn2index = {f:i for i,f in enumerate(fns)}
            # fn = st.selectbox('Select Function', fn2index['__init__'], key=f'fn.{prefix}')
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

                    module_name = module
                    module = c.module(module)

                    for tag in tags:
                        st.write(f'Registering {module_name} with tag {tag}, {kwargs}')
                        response = module.register(tag=tag, subnet= self.subnet, stake=stake)
                        st.write(response)
                except Exception as e:
                    response = {'success': False, 'message': str(e)}
                    raise e
                if response['success']:
                    st.success('Module Registered')
                else:
                    st.error(response['message'])
                



    def modules_dashboard(self):
        import pandas as pd
        # search  for all of the modules with yaml files. Format of the file
        search = st.text_input('Search', '')
        df = None
        
        self.searched_modules = [m for m in self.modules if search in m['name'] or search == '']
        df = pd.DataFrame(self.searched_modules)
        if len(df) == 0:
            st.error(f'{search} does not exist {c.emoji("laughing")}')
            return
        else:
            st.success(f'{c.emoji("dank")} {len(df)} modules found with {search} in the name {c.emoji("dank")}')
            del df['stake_from']
            st.write(df)

        
            with st.expander('Historam'):
                key = st.selectbox('Select Key', ['incentive',  'dividends', 'emission'], 0)
                
                self.st.run(df)
                fig = px.histogram(
                    x = df[key].to_list(),
                )

                st.plotly_chart(fig)


if __name__ == '__main__':
    Dashboard.run()
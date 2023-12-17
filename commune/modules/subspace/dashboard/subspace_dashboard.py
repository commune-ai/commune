import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px
import streamlit as st


class SubspaceDashboard(c.Module):
    
    def __init__(self, state=None, key=None): 

        t1 = c.time()
        t2 = c.time()
        time = t2 - t1
        st.write(f'Loaded State in {time} seconds')
        self.select_key()
        self.load_state()
        self.select_netuid()


        if key != None:
            self.key = key

        
        # convert into metrics

    def select_key(self):
        import streamlit as st
        keys = c.keys()

        key2index = {k:i for i,k in enumerate(keys)}
        cols = st.columns([2,2])
        self.key = cols[0].selectbox('Select Key', keys, key2index['module'], key='key.sidebar')
        key_address = self.key.ss58_address
        cols[1].write('\n')
        cols[1].code(key_address)
        return self.key
    def select_netuid(self):
        import streamlit as st
        keys = c.keys()
        subnets = self.subnets
        name2subnet = {s['name']:s for s in subnets}
        name2idx = {s['name']:i for i,s in enumerate(subnets)}
        subnet_names = list(name2subnet.keys())
        subnet = st.selectbox('Select Subnet', subnet_names, 0, key='subnet.sidebar')
        self.netuid = name2idx[subnet]
        return self.key
        
    def transfer_dashboard(self):
        with st.expander('Transfer', expanded=False):
            cols = st.columns(2)
            amount = cols[0].number_input('amount', 0.0, 10000000.0, 0.0, 0.1)
            to_address = cols[1].text_input('dest (s) : use , for multiple transfers', '')
            multi_transfer = False

            

            if ',' in to_address:
                multi_transfer = True
                to_addresses = [a.strip() for a in to_address.split(',')]

            transfer_button = st.button('Transfer')
            if transfer_button:

                if multi_transfer:
                    kwargs = {
                        'destinations': to_addresses,
                        'amounts': amount,
                        'key': self.key,
                    }
                    st.write(kwargs)
                    response = c.multitransfer(**kwargs)

                else:

                    kwargs = {
                        'dest': to_address,
                        'amount': amount,
                        'key': self.key,
                    }
                    response = c.transfer(**kwargs)

                st.write(response)


    def network_dashboard(self):
        modules = self.modules
        modules = self.key_info['stake_to']
        # st.write(modules)


    def stake_dashboard(self):
        
        with st.expander('Stake', expanded=False):
            cols = st.columns(2)
            staked_modules = self.module_names
            default_staked_modules = []
            modules = st.multiselect('Modules', staked_modules, default_staked_modules)

            amounts = st.number_input('Stake Amount', value=0.0,  max_value=1000000000000.0, min_value=0.0 ) # format with the value of the balance            
            
            st.write(f'You have {len(modules)} ready to STAKE for a total of {amounts * len(modules)} ')
            
            stake_button = st.button('STAKE')

            if stake_button:
                kwargs = {
                    'amounts': amounts,
                    'modules': modules,
                    'key': self.key,
                }

                response = c.multistake(**kwargs)
                st.write(response)


    def unstake_dashboard(self):
        with st.expander('Unstake', expanded=False):
            modules = list(self.key_info['stake_to'].keys())
            cols = st.columns(4)

            amounts = cols[0].number_input('Unstake Amount',0)
            default_modules = [k for k,v in self.key_info['stake_to'].items() if v > amounts]
            default_values = [v for k,v in self.key_info['stake_to'].items() if v > amounts]
            search = cols[1].text_input('Search', '', key='search.unstake')
            n = len(default_modules)
            st.write(f'You have {n} modules staked')

            n = cols[3].number_input('Number of Modules', 1, n, 10, 1, key=f'n.unstake')
            if search != '':
                modules = [m for m in modules if search in m]
                default_modules = [m for m in default_modules if search in m]
            modules = st.multiselect('Module', modules, default_modules[:n])
            total_stake_amount = amounts * len(modules)
            
            st.write(f'You have {len(modules)} ready to UNSTAKE for a total of {total_stake_amount} ')

            unstake_button = st.button('UNSTAKE')
            if unstake_button:
                kwargs = {
                    'amounts': amounts,
                    'modules': modules,
                    'key': self.key,
                }
                response = c.multiunstake(**kwargs)
                st.write(response)





    def archive_dashboard(self):
        # self.register_dashboard(expanded=False)
        netuid = 0 
        archive_history = c.archive_history(lookback_hours=24, n=100, update=True)
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
        # st.write(state)f

        st.write(fig)
        # options = ['emission', 'incentive', 'dividends', 'stake']
        # y = st.selectbox('Select Columns', options, 0)
        # # filter by stake > 1000

        # df = df[df['stake'] > 10**9]
        # histogram = px.histogram(df, x=y, title='My Modules')

        # st.write(histogram)

    # def 
    #     with st.expander('Modules', expanded=False):
    #         import pandas as pd
    #         # search  for all of the modules with yaml files. Format of the file
    #         search = st.text_input('Search', '')
    #         df = None
    #         self.modules = self.state['modules'][self.netuid]

        
    #         self.searched_modules = [m for m in self.modules if search in m['name'] or search == '']
    #         df = pd.DataFrame(self.searched_modules)
    #         if len(df) == 0:
    #             st.error(f'{search} does not exist {c.emoji("laughing")}')
    #             return
    #         else:
    #             st.success(f'{c.emoji("dank")} {len(df)} modules found with {search} in the name {c.emoji("dank")}')
    #             del df['stake_from']
    #             st.write(df)
    #             # with st.expander('Historam'):
    #             #     key = st.selectbox('Select Key', ['incentive',  'dividends', 'emission'], 0)
                    
    #             #     self.st.run(df)
    #             #     fig = px.histogram(
    #             #         x = df[key].to_list(),
    #             #     )

    #             #     st.plotly_chart(fig)

    def register_dashboard(self, expanded=True, prefix= None, form = True ):


        if expanded : 
            with st.expander('Register', expanded=False):
                return self.register_dashboard(prefix=prefix, expanded=False)
        modules = c.modules(prefix)
        cols = st.columns([2,2,2])

        with st.form(key='register'):
            module  = cols[0].selectbox('Select A Module', modules, 0)
            tag = cols[1].text_input('tag', c.random_word(n=1), key=f'tag.register')
            stake = cols[2].number_input('stake', 0.0, 10000000.0, 0.1, key=f'stake.{prefix}.register')
            n = st.number_input('Number of Replicas', 1, 30, 1, 1, key=f'n.{prefix}.register')
            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            register = st.form_submit_button('Register')
            # fn = st.selectbox('Select Function', fn2index['__init__'], key=f'fn.{prefix}')
            kwargs = self.function2streamlit(module=module, fn='__init__', salt='register')

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
                    e = c.detailed_error(e)
                    response = {'success': False, 'message': e}
                    raise e
                if response['success']:
                    st.success('Module Registered')
                else:
                    st.error(response['message'])

    
    def key_info_dashboard(self, expander = True):
        if expander:
            with st.expander('Key Info', expanded=False):
                return self.key_info_dashboard(expander=False)

        # pie map of stake
        st.write('# Wallet')
        st.write('ss58_address')

        st.code( self.key.ss58_address)
        cols = st.columns([2,2])
        cols[0].metric('Balance', int(self.key_info['balance']))
        cols[1].metric('Stake', int(self.key_info['stake']))
            
        

        values = list(self.key_info['stake_to'].values())
        labels = list(self.key_info['stake_to'].keys())

        sorted_labels_and_values = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels = [l for l,v in sorted_labels_and_values]
        values = [v for l,v in sorted_labels_and_values]


        modules = self.modules
        my_modules = [m for m in modules if m['name'] in labels]
        daily_reward = sum([m['emission'] for m in my_modules]) * 108
        st.write('### Daily Reward', daily_reward)        

        st.write('## My Staked Modules')
        for m in my_modules:
            m.pop('stake_from')
        df = pd.DataFrame(my_modules)
        st.write(df)

    @classmethod
    def dashboard(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
       


        # bar chat of staked modules
        self.network_dashboard()
        # self.key_info_dashboard() 
        self.stake_dashboard()
        self.unstake_dashboard()
        self.transfer_dashboard()
        self.register_dashboard()
        # if tokenomics:
        #     c.module('subspace.tokenomics').dashboard()




SubspaceDashboard.run(__name__)
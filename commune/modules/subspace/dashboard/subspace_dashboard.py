import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px
import streamlit as st


class SubspaceDashboard(c.Module):
    
    def __init__(self, state=None, key=None): 

        self.load_state()
        if key != None:
            self.key = key

        
        # convert into metrics
        
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


    def stake_dashboard(self):
        
        with st.expander('Stake', expanded=False):

            cols = st.columns(2)
            entire_balance = cols[0].checkbox('Entire Balance', key='entire_balance')
            my_staked_button = cols[1].checkbox('My Staked Modules Only', key='my_staked')
            cols = st.columns(3)
            search = cols[0].text_input('Search', '', key='subsapce.search.stake')

    
            staked_modules = self.module_names
            default_staked_modules = []
            if my_staked_button:
                staked_modules = list(self.key_info['stake_to'].keys())
                default_staked_modules = staked_modules
            if search != '':
                default_staked_modules = [m for m in staked_modules if search in m]

            n = cols[2].number_input('Number of Modules', 1, len(staked_modules), 1, 1, key=f'n.stake')
            modules = st.multiselect('Modules', staked_modules, default_staked_modules[:n])
            # resolve amoujnts
            if entire_balance:
                try:
                    default_amount = c.balance(self.key.ss58_address)  / len(modules)
                except Exception as e:
                    st.error('No Internet Connection to Substrate')
                    default_amount = 0.0
            else:
                default_amount = 0.0
            amounts = cols[1].number_input('Stake Amount', value=default_amount,  max_value=1000000000000.0, min_value=0.0 ) # format with the value of the balance            
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
            cols = st.columns(3)

            amounts = cols[0].number_input('Unstake Amount',0)
            default_modules = [k for k,v in self.key_info['stake_to'].items() if v > amounts]
            default_values = [v for k,v in self.key_info['stake_to'].items() if v > amounts]
            search = cols[1].text_input('Search', '', key='search.unstake')
            n = cols[2].number_input('Number of Modules', 1, 10000000, 1, 1, key=f'n.unstake')
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
            tag = cols[1].text_input('tag', c.random_word(n=2), key=f'tag.register')
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

    @classmethod
    def dashboard(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
       
        # pie map of stake
        st.write('# Wallet')

        with st.expander('Key Info', expanded=False):
            st.write('ss58_address')
            cols = st.columns([2,2,2])
            cols[0].code( self.key.ss58_address)
            cols[1].metric('Balance', int(self.key_info['balance']))
            cols[2].metric('Stake', int(self.key_info['stake']))
            
    
            values = list(self.key_info['stake_to'].values())
            labels = list(self.key_info['stake_to'].keys())

            fig = c.module('plotly').treemap(values=values, labels=labels, title=None)
            # increase the width of the plot
            fig.update_layout(width=900, height=750)
            st.plotly_chart(fig)

        # bar chat of staked modules


        self.stake_dashboard()
        self.unstake_dashboard()
        self.transfer_dashboard()
        self.register_dashboard()







SubspaceDashboard.run(__name__)
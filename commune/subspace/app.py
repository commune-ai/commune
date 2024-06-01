import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px
import streamlit as st

css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)

class SubspaceDashboard(c.Module):
    
    def __init__(self, state=None, key=None): 
        self.st = c.module('streamlit')()
        self.st.load_style()
        t1 = c.time()
        t2 = c.time()
        time = t2 - t1
        with st.sidebar:
            self.select_key()
        self.load_state()
        self.subnet_dashboard()

        


        # self.my_info_dashboard(expander=False)



        if key != None:
            self.key = key
        
        # convert into metrics

        # self.archive_dashboard()


    def subnet_dashboard(self):

        subnet_params = pd.DataFrame(self.state['subnets'])
        st.write(subnet_params)



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
        c.print(subnet_names)
        subnet_name = st.selectbox('Select Subnet', subnet_names, 0, key='subnet.sidebar')
        self.netuid = name2idx[subnet_name]
        self.subnet = name2subnet[subnet_name]
        self.modules = self.state['modules'][self.netuid]

        return self.key
        
    def transfer_dashboard(self):
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
                response = c.transfer_many(**kwargs)

            else:

                kwargs = {
                    'dest': to_address,
                    'amount': amount,
                    'key': self.key,
                }
                response = c.transfer(**kwargs)

            st.write(response)



    def stake_dashboard(self):
        
        cols = st.columns(2)
        staked_modules = list(self.namespace.keys())
        default_staked_modules = []
        modules = st.multiselect('Modules', staked_modules, default_staked_modules)
        balance = self.key_info['balance']
        amounts = st.slider('Stake Amount', 0.0, balance, balance, 0.1)
        
        st.write(f'You have {len(modules)} ready to STAKE for a total of {amounts * len(modules)} ')
        
        stake_button = st.button('STAKE')

        if stake_button:
            kwargs = {
                'amounts': amounts,
                'modules': modules,
                'key': self.key,
                'netuid': self.netuid,
            }

            response = c.stake_many(**kwargs)
            st.write(response)

    def staking_dashboard(self):
        # st.write(self.key_info)
        # turn key_info into a dataframe
        # st.write(self.key_info)
        self.staking_plot()
        with st.expander('Staking', False):
            self.stake_dashboard()
        with st.expander('Unstaking', False):
            self.unstake_dashboard()


    def staking_plot(self):
        stake_to = self.key_info['stake_to']
        df = pd.DataFrame(stake_to.items(), columns=['module', 'stake'])




        if len(df) > 0 : 
            df = pd.DataFrame(stake_to.items(), columns=['module', 'stake'])
            df.sort_values('stake', inplace=True, ascending=False)
            # get a bar chart of the stake
            fig = px.bar(df, x='module', y='stake', title='Stake')

            st.plotly_chart(fig)
        else:
            st.error(f'You are not staked to any modules {c.emoji("laughing")}')


    def unstake_dashboard(self):
        modules = list(self.key_info['stake_to'].keys())

        default_modules = [k for k,v in self.key_info['stake_to'].items()]
        default_values = [v for k,v in self.key_info['stake_to'].items()]
        search = st.text_input('Search', '', key='search.unstake')
        n = len(default_modules)
        st.write(f'You have {n} modules staked')
        unstake_percentage = st.slider('Unstake %', 0.0, 100.0) # format with the value of the balance


        n = st.slider('Number of Modules', 1, n, 10, 1, key=f'n.unstake')
        if search != '':
            default_modules = [m for m in default_modules if search in m]
        default_modules = default_modules[:n]
        total_stake = sum([self.key_info['stake_to'][m] for m in default_modules])
        total_stake_amount = total_stake * (unstake_percentage / 100)
        st.write(f'You have {len(default_modules)} ready to UNSTAKE for a total of {total_stake_amount} ')
       
    
        # module_names = st.multiselect('Module', default_modules, default_modules )
        module_names = default_modules
        df = [{'module': m, 'stake': self.key_info['stake_to'][m]} for m in module_names]
        # add percentage unstaked to the dataframe
        for i in range(len(df)):
            df[i]['unstake'] = df[i]['stake'] * (unstake_percentage / 100)
            df[i]['stake'] = df[i]['stake'] - df[i]['unstake']
        df = pd.DataFrame(df)
        if len(df) > 0:
            df.sort_values('stake', inplace=True, ascending=False)
            # get a bar chart of the stake
            fig = px.bar(df, x='module', y='stake', title='Unstake vs Stake')
            fig.add_bar(x=df['module'], y=df['unstake'], name='Unstake')
            st.plotly_chart(fig)
            
        with st.form('unstake'):
            total_stake_amount = unstake_percentage * len(modules)
        
            st.write(f'You have {len(module_names)} ready to UNSTAKE for a total of {total_stake_amount} ')

            unstake_button = st.form_submit_button('UNSTAKE')
            if unstake_button:
                kwargs = {
                    'amounts': unstake_percentage,
                    'modules': module_names,
                    'key': self.key,
                    'netuid': self.netuid,
                }
                response = c.unstake_many(**kwargs)
                st.write(response)




    def register_dashboard(self, expanded=True, prefix= None, form = True ):


        cols = st.columns([2,2,2])

        modules = c.modules(prefix)
        module  = cols[0].selectbox('Select A Module' , modules, 0)
        subnet = cols[1].text_input('Subnet', self.subnet['name'])
        with st.form(key='register'):
            tag = cols[1].text_input('tag', c.random_word(n=1), key=f'tag.register')
            stake = cols[2].number_input('stake', 0.0, 10000000.0, 0.1, key=f'stake.{prefix}.register')
            n = st.number_input('Number of Replicas', 1, 30, 1, 1, key=f'n.{prefix}.register')
            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            # fn = st.selectbox('Select Function', fn2index['__init__'], key=f'fn.{prefix}')
            kwargs = self.function2streamlit(module=module, fn='__init__', salt='register')
            register = st.form_submit_button('Register')

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
                    try:
                        module = c.module(module)
                    except Exception as e:
                        e = c.detailed_error(e)
                        response = {'success': False, 'message': e}
                        st.error(response)
                        return

                    for tag in tags:
                        try:
                            response = module.register(tag=tag, subnet= subnet, stake=stake)
                            st.write(response)
                        except Exception as e:
                            e = c.detailed_error(e)
                            response = {'success': False, 'message': e}
                            st.error(response)
                except Exception as e:
                    e = c.detailed_error(e)
                    response = {'success': False, 'message': e}
                    raise e
                if response['success']:
                    st.success('Module Registered')
                else:
                    st.error(response['message'])

    
    def my_info_dashboard(self, expander = True):
            
        # pie map of stake

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

       

        fns = cls.fns()
        fns = ['_'.join(f.split('_')[:-1]).replace('_',' ').lower() for f in fns if f.endswith('_dashboard')]
        fns = ['staking', 'register', 'transfer', 'my info' , 'subnet info']
        tabs = st.tabs(fns)


        for i, fn in enumerate(fns):
            with tabs[i]:
                fn_name = fn.lower().replace(' ', '_') + '_dashboard'
                if not hasattr(self, fn_name):
                    st.error(f'{fn_name} not found in {self.__class__.__name__}')
                else:
                    fn = getattr(self, fn_name)
                    fn()



        # if tokenomics:
        #     c.module('subspace.tokenomics').dashboard()



    def load_state(self, update:bool=False, netuid=0, network='main', state=None, _self = None):
        
        if _self != None:
            self = _self
        
        import streamlit as st
        
        self.key = c.get_key()

        t = c.timer()
        @st.cache_data(ttl=60*60*24, show_spinner=False)
        def get_state():
            subspace = c.module('subspace')()
            state =  subspace.state_dict(update=update)
            return state
        
        if state == None:
            state = get_state()
        self.state =  state
        self.netuid = 0
        self.subnets = self.state['subnets']

        with st.sidebar:
            self.select_netuid()




        self.modules = self.state['modules'][self.netuid]

        self.name2module = {m['name']: m for m in self.modules}
        # c.print(self.modules)
        self.name2key = {k['name']: k['key'] for k in self.modules}
        self.key2name = {k['key']: k['name'] for k in self.modules}

        self.keys  = c.keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}

        self.namespace = {m['name']: m['address'] for m in self.modules}

        for i, m in enumerate(self.modules):
            self.modules[i]['stake'] = m['stake']/1e9
            self.modules[i]['emission'] = m['emission']/1e9
        # st.write(self.state['stake_to'][self.netuid].get(self.key.ss58_address))
        # st.write(self.state['stake_to'][self.netuid])
        stake_to = self.state['stake_to'][self.netuid].get(self.key.ss58_address)
        self.key_info = {
            'key': self.key.ss58_address,
            'balance': self.state['balances'].get(self.key.ss58_address,0),
            'stake_to': self.state['stake_to'][self.netuid].get(self.key.ss58_address,{}),
            'stake': sum([v[1] for v in stake_to]) if stake_to != None else {},
        }

        self.key_info['balance']  = self.key_info['balance']/1e9
        self.key_info['stake_to'] = {k:v/1e9 for k,v in self.key_info['stake_to']}
        self.key_info['stake'] = sum([v for k,v in self.key_info['stake_to'].items()])
        # convert keys to names 
        for k in ['stake_to']:
            self.key_info[k] = {self.key2name[k]: v for k,v in self.key_info[k].items() if k in self.key2name}
       
        balances = self.state['balances']
        self.total_balance = sum(balances.values())/1e9

    def subnet_info_dashboard(self):
        subnet_df = pd.DataFrame(self.subnets)
        self.subnet_info = {}
        self.subnet_info['params'] = self.state['subnets'][self.netuid]
        with st.expander('Subnet Info', expanded=False):
            self.subnet_info['n'] = len(self.modules)
            self.subnet_info['total_stake'] = sum([m['stake'] for m in self.modules])
            st.write(self.subnet_info)   


    


    # def archive_dashboard(self):
    #     # self.register_dashboard(expanded=False)
    #     netuid = 0 
    #     archive_history = c.archive_history(lookback_hours=24, n=100, update=True)
    #     df = c.df(archive_history[1:])
    #     df['block'] = df['block'].astype(int)


    #     df['dt'] = pd.to_datetime(df['dt'])
    #     df.sort_values('block', inplace=True)
    #     df.reset_index(inplace=True)
    #     st.write(df)
    #     # df= df[df['market_cap'] < 1e9]


    #     fig = px.line(df, x='block', y='market_cap', title='Archive History')

    #     block2path= {b:df['path'][i] for i,b in enumerate(df['block'])}
    #     blocks = list(block2path.keys())
    #     paths = list(block2path.values())
    #     block = st.selectbox('Block', blocks, index=0)
    #     path = block2path[block]
    #     state = c.get(path)
    #     modules = state['modules'][netuid]
    #     for i in range(len(modules)):
    #         for k in ['stake_to', 'stake_from', 'key', 'address']:
    #             del modules[i][k]
    #         for k in ['emission', 'stake', 'balance']:
    #             modules[i][k] = modules[i][k]/1e9
    #     df = pd.DataFrame(modules)

    #     st.write(df)
    #     subnet_df = pd.DataFrame(state['subnets'])
    #     st.write(subnet_df)
    #     # st.write(state)f

    #     st.write(fig)
    #     # options = ['emission', 'incentive', 'dividends', 'stake']
    #     # y = st.selectbox('Select Columns', options, 0)
    #     # # filter by stake > 1000

    #     # df = df[df['stake'] > 10**9]
    #     # histogram = px.histogram(df, x=y, title='My Modules')

    #     # st.write(histogram)

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


SubspaceDashboard.run(__name__)
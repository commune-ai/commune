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
logo_path = __file__.replace('dashboard.py', 'commune_logo.png')


class Wallet(c.Module):



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

    def module_dashboard(self):
        search = st.text_input('Search Namespace', '', key='search.namespace')
        df = []
        df = []
        for m in self.modules:
            if search in m['name'] or search in m['address'] or search in m['key']:
                m.pop('stake_from', None)
                m.pop('stake_to', None)
                df.append(m)
        df = c.df(df)
        n = len(df)
        st.write(f'**{n}** modules with **{search}** in it')
        st.write(df)
        playground = st.checkbox('Playground')
        if playground:
            self.playground()

    def stake_dashboard(self):
        
        cols = st.columns(2)
        staked_modules = list(self.namespace.keys())
        default_staked_modules = staked_modules[:10]
        cols = st.columns([4,3])
        balance = self.key_info['balance']


        with cols[1]:
            stake_ratio = st.slider('Ratio of Balance', 0.0, 1.0, 0.5, 0.01, key='stake.ratio')
            amounts = int(st.number_input('Stake Across Modules', 0.0, balance, balance*stake_ratio, 0.1))
            modules = st.multiselect('Modules', staked_modules, default_staked_modules)
            stake_button = st.button('ADD STAKE')

        with cols[0]:
            df = []
            module2amount = {m:amounts/len(modules) for m in modules}
            amounts = list(module2amount.values())
            total_stake = sum(amounts)

            if len(modules) == 0:
                for k,v in self.key_info['stake_to'].items():
                    df.append({'module': k, 'stake': v, 'added_stake':0})
                    if k in module2amount:
                        df[-1]['added_stake'] = amounts
                        module2amount.pop(k)
            else:
                for k,v in self.key_info['stake_to'].items():
                    if k in module2amount:
                        df.append({'module': k, 'stake': v , 'add_stake':module2amount[k]})
                    else:
                        df.append({'module': k, 'stake': 0, 'add_stake':v})
            
            df = pd.DataFrame(df)
    
            if len(df) > 0:
                
                df.sort_values('stake', inplace=True, ascending=False)
                title = f'Staking {total_stake:.2f} to {len(modules)} modules'
                import plotly.graph_objects as go
                df['color'] = 'green'
                fig = px.bar(df, x='module', y='add_stake', title=title, color='color', color_discrete_map={'green':'green'})
                # TEXT ON THE BARS
                fig.update_traces(texttemplate='%{y:.2f}', textposition='inside')
                # remove x axis labels

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error('You are not staked to any modules')
        
        

        if stake_button:
            kwargs = {
                'amounts': amounts,
                'modules': modules,
                'key': self.key,
                'netuid': self.netuid,
                'network': self.network,
            }

            response = c.stake_many(**kwargs)
            st.write(response)


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
        
        cols = st.columns(2)
        staked_modules = list(self.namespace.keys())
        default_staked_modules = list(self.key_info['stake_to'].keys())[:10]
        cols = st.columns([4,3])
        balance = self.key_info['balance']

        with cols[1]:
            stake_button = st.button('REMOVE STAKE')
            modules = st.multiselect('Modules', staked_modules, default_staked_modules, key='modules.unstake')
            total_amount = sum([self.key_info['stake_to'][m] for m in modules])
            unstake_ratio = st.slider('Unstake Ratio', 0.0, 1.0, 0.5, 0.01, key='unstake.ratio')
            unstake_amount = st.number_input('Unstake Amount', 0.0, float(total_amount), float(total_amount*unstake_ratio), 0.1, key='unstake.amount')

            unstake_ratio = unstake_amount/(total_amount + 1e-9)


        with cols[0]:
            df = []
            module2amount = {m: self.key_info['stake_to'][m] * unstake_ratio for m in modules}
            amounts = list(module2amount.values())
            if len(modules) == 0:
                for k,v in self.key_info['stake_to'].items():
                    df.append({'module': k, 'stake': v, 'unstake':0})
            else:
                for k,v in self.key_info['stake_to'].items():
                    if k in module2amount:
                        df.append({'module': k, 'stake': v - module2amount[k], 'unstake':module2amount[k]})
            
            df = pd.DataFrame(df)

            if len(df) > 0:
                df['color'] = 'unstaking'
                df.sort_values('stake', inplace=True, ascending=False)
                title = f'Removing {unstake_amount:.2f} Stake from {len(modules)} modules'
                fig = px.bar(df, x='module', y='unstake', title=title, color='color', color_discrete_map={'unstaking':'red'})
                df['color'] = 'staking'
                fig.add_bar(x=df['module'], y=df['stake'], marker_color='green', name='staking')
                fig.update_traces(texttemplate='%{y:.2f}', textposition='inside')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error('You are not staked to any modules')
        
        
        

        if stake_button:
            kwargs = {
                'amounts': amounts,
                'modules': modules,
                'key': self.key,
                'netuid': self.netuid,
            }

            response = c.unstake_many(**kwargs)
            st.write(response)


    def register_dashboard(self, expanded=True, prefix= None, form = True ):


        cols = st.columns([2,2])

        modules = c.modules(prefix)
        module  = cols[0].selectbox('Select A Module' , modules, 0)
        tag = cols[1].text_input('tag', c.random_word(n=1), key=f'tag.register')
        cols = st.columns([2,2])
        subnet = cols[0].text_input('Subnet', self.subnet['name'])
        stake = cols[1].number_input('stake', 0.0, 10000000.0, 0.1, key=f'stake.{prefix}.register')

        with st.form(key='register'):
            # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
            # fn = st.selectbox('Select Function', fn2index['__init__'], key=f'fn.{prefix}')
            with st.expander('INIT KWARGS', expanded=True):
                st.write(module)
                kwargs = self.function2streamlit(module=module, fn='__init__')
            n = st.slider('Replicas', 1, 30, 1, 1, key=f'n.{prefix}.register')

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

    
    def my_info(self, expander = True):
        st.write('My Public Address')
        st.code(self.key.ss58_address)

        modules = self.modules

        values = list(self.key_info['stake_to'].values())
        labels = list(self.key_info['stake_to'].keys())
        sorted_labels_and_values = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels = [l for l,v in sorted_labels_and_values]
        values = [v for l,v in sorted_labels_and_values]

        my_modules = [m for m in modules if m['name'] in labels]

        my_info_df = pd.DataFrame(self.key_info)
        for m in my_modules:
            m.pop('stake_from', None)
        
        
        
        df = pd.DataFrame(my_modules)


    @classmethod
    def dashboard(cls, key = None):
        self = cls()
        self.st = c.module('streamlit')()
    
        c.load_style()
        self.logo_url = "https://github.com/commune-ai/commune/blob/librevo/commune/modules/dashboard/commune_logo.gif?raw=true"
        st.markdown(f"![Alt Text]({self.logo_url}), width=10, height=10")
        with st.sidebar:
            self.select_network()
    
            self.select_key()

        fns = self.fns()
        dash_fns = [f for f in fns if f.endswith('_dashboard')]
        options = [f.replace('_dashboard', '') for f in dash_fns]


        # self.my_info(expander=False) 


        tabs = st.tabs(options)

        for i, tab in enumerate(tabs):
            dash_fn = dash_fns[i]
            with tab:
                getattr(self, dash_fn)()

    def select_key(self):
        keys = c.keys()
        key2index = {k:i for i,k in enumerate(keys)}
        self.key = st.selectbox('Select Key', keys, key2index['module'], key='key.sidebar')
        ss58_address = self.key.ss58_address

        return self.key


    def select_network(self, update:bool=False, netuid=0, network='main', state=None, _self = None):
        
        self = _self or self
        
        import streamlit as st
        
        subspace = c.module('subspace')()
        networks = ['main', 'test']
        
        self.network = st.selectbox('Select Network', networks, 0, key='network')
        update = st.button('Update')

        subnets = subspace.subnet_params(netuid='all', network=self.network, update=update)
        name2subnet = {s['name']:s for s in subnets}
        subnet_names = [s['name'] for s in subnets]
        name2netuid = {s['name']:i for i, s in enumerate(subnets)}
        subnet_name = st.selectbox('Select Subnet', subnet_names, 0, key='subnet')
        netuid = name2netuid[subnet_name]
        self.netuid = netuid
        self.subnet = name2subnet[subnet_name]
        self.subnets = subnets
        self.namespace = {m['name']: m['address'] for m in self.modules}

Wallet.run(__name__)
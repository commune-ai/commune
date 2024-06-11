import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px

css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)
logo_path = __file__.replace('dashboard.py', 'commune_logo.png')

class Dashboard(c.Module):
    def __init__(self, stfate=None, key=None): 
        self.subnet = None
        self.st = c.module('streamlit')()
        self.st.load_style()
        with st.sidebar:
            st.title(f'COMMUNE')
        with st.sidebar:
            cols = st.columns([2,2])
            self.select_key()



        self.logo_url = "https://github.com/commune-ai/commune/blob/librevo/commune/modules/dashboard/commune_logo.gif?raw=true"
        # st.markdown(f"![Alt Text]({self.logo_url}), width=10, height=10")





        # self.my_info_dashboard(expander=False)



        if key != None:
            self.key = key
        
        # convert into metrics

        # self.archive_dashboard()




    def select_key(self):
        keys = c.keys()
        key2index = {k:i for i,k in enumerate(keys)}
        self.key = st.selectbox('Select Key', keys, key2index['module'], key='key.sidebar')
        self.key_info = {
            'key': self.key.ss58_address,
        }
        return self.key

        
    def transfer_dashboard(self):
        cols = st.columns(2)
        amount =  cols[0].number_input('amount', 0.0, 10000000.0, 0.0, 0.1)
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
            df.sort_values('stake', inplace=True, ascending=False)
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
            stake = self.key_info['stake']
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
                kwargs = self.function2streamlit(module=module, fn='__init__', salt='register')
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


    @classmethod
    def dashboard(cls, *args, **kwargs):
        
    
        self = cls(*args, **kwargs)

        fns = cls.fns()
        fns = ['_'.join(f.split('_')[:-1]).replace('_',' ').lower() for f in fns if f.endswith('_dashboard')]
        fns = ['wallet' , 'explorer']
        options = c.modules()
        page = st.sidebar.selectbox('Select Page', options, 0, key='page')

        st.sidebar.markdown('---')
        st.sidebar.markdown(f'**{page}**')
        module = page.lower()
        module = c.module(page)
        fn_name = 'dashboard'
        fn = getattr(module, fn_name)
        schema = module.schema().get(fn_name)
        if 'key' in schema['input']:
            fn(key=self.key)
        else:
            fn()

        # if tokenomics:
        #     c.module('subspace.tokenomics').dashboard()



Dashboard.run(__name__)

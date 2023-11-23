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

    
    def key_info(self, expander = True):
        if expander:
            with st.expander('Key Info', expanded=False):
                return self.key_info(expander=False)

        # pie map of stake
        st.write('# Wallet')
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

    @classmethod
    def dashboard(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
       

        # bar chat of staked modules

        self.tokenomics()

        self.stake_dashboard()
        self.unstake_dashboard()
        self.transfer_dashboard()
        self.register_dashboard()


    def emission_schedule(self, 
                          days = 10000, 
                        starting_emission = 0, 
                        emission_per_day = 1,
                        emission_per_halving = 250_000_000,
                        burn_rate = 0.5,
                        n = 1000,
                        dividend_rate = 0.5,
                        ):
        emission_per_day = emission_per_day
        halving_factor = 0.5

        state = c.munch({
            'emission_per_day': [],
            'burned_emission_per_day': [],
            'total_emission': [],
            'halving_factor': [],
            'day': [],
            'burn_price_per_day': [],
            'dividends_per_token': [],
            'required_stake_to_cover_burn': [],
        })
        total_emission = starting_emission
        
        for day in range(days):
            halvings = total_emission // emission_per_halving
            curring_halving_factor = halving_factor ** halvings
            current_emission_per_day = emission_per_day * curring_halving_factor

            current_burned_emission_per_day = current_emission_per_day * burn_rate
        
            daily_dividends_per_token = current_emission_per_day * (1 / total_emission) * dividend_rate
            current_burned_emission_per_module = current_burned_emission_per_day / n
            state.required_stake_to_cover_burn.append(current_burned_emission_per_module / daily_dividends_per_token)
            state.dividends_per_token.append(daily_dividends_per_token)


            state.burn_price_per_day.append(current_burned_emission_per_module)

            
            current_emission_per_day -= current_burned_emission_per_day

            total_emission += current_emission_per_day
            state.total_emission.append(total_emission)
            state.emission_per_day.append(current_emission_per_day)
            state.burned_emission_per_day.append(current_burned_emission_per_day)
            state.day.append(day)
            state.halving_factor.append(curring_halving_factor)


            # calculate the expected apy
            # 1. calculate the total supply
            # 2. calculate the total stake
            # 3. calculate the total stake / total supply



        state = c.munch2dict(state)
    
        df = c.df(state)
        df['day'] = pd.to_datetime(df['day'], unit='D', origin='2023-11-23')

        return df
    
    def tokenomics(self): 
        st.write('# Tokenomics')
        cols = st.columns(2)

        emission_per_day = cols[0].number_input('Emission Per Day', 0, 1_000_000, 250_000, 1)
        
        starting_emission = cols[1].number_input('Starting Emission', 0, 100_000_000_000, 60_000_000, 1) 

        days = st.slider('Days', 1, 3_000, 800, 1)

        n = st.number_input('Number of Modules', 1, 1000000, 8400, 1, key=f'n.modules')

        burn_rate = st.slider('Burn Rate', 0.0, 1.0, 0.0, 0.01, key=f'burn.rate')

        dividend_rate = st.slider('Dividend Rate', 0.0, 1.0, 0.5, 0.01, key=f'dividend.rate')

        burned_emission_per_day = emission_per_day * burn_rate
        block_time = 8
        tempo = 100
        seconds_per_day =  24 * 60 * 60 
        seconds_per_year = 365 * seconds_per_day
        blocks_per_day = seconds_per_day / block_time
        blocks_per_year = seconds_per_year / block_time
        emission_per_block = emission_per_day / blocks_per_day
        emission_per_year = blocks_per_year * emission_per_block

        df = self.emission_schedule(days=days, 
                                    starting_emission=starting_emission, 
                                    emission_per_day=emission_per_day, 
                                    emission_per_halving=250_000_000,
                                    burn_rate = burn_rate,
                                    n=n,
                                    dividend_rate=dividend_rate


                                    )
        

        with st.expander('Dividends Calculator', expanded=True):

            my_stake = st.number_input('My Stake', 0, 1000000000000, 1000, 1, key=f'my.stake')
            final_stake = my_stake
            stake_over_time = [my_stake]
            appreciation = []


            for i in range(len(df['dividends_per_token'])):
                burn_price_per_day = df['burn_price_per_day'][i]
                stake_over_time.append(df['dividends_per_token'][i] * stake_over_time[-1] + stake_over_time[-1] - burn_price_per_day)
                appreciation.append(stake_over_time[-1] / stake_over_time[0])
            stake_over_time = stake_over_time[1:]


            
            df['stake_over_time'] = stake_over_time
            df['appreciation'] = appreciation

            # make subplots 
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            figure = make_subplots(specs=[[{"secondary_y": True}]])
            # green if positive red if negative
            figure.add_trace(
                go.Scatter(x=df['day'], y=df['stake_over_time'], name='stake_over_time'),
                secondary_y=True,
            )
            figure.add_trace(
                go.Scatter(x=df['day'], y=df['appreciation'], name='appreciation'),
                secondary_y=False,
            )
            
            # dual axis
            # do this with red and green if the line is positive or negative
            st.plotly_chart(figure)




        st.markdown(f"""
         Emission per Block: {emission_per_block} \n
         Emission per Day: {emission_per_day} \n
         Emission per Year: {emission_per_year} \n
         Emission per Tempo: {emission_per_day * tempo} \n
         Emission per Tempo per Year: {emission_per_year * tempo}
        """)

        st.write(df)


        # convert day into datetime from now


        y_options = df.columns
        x_options = df.columns

        y = st.selectbox('Select Y', y_options, 0)
        x = st.selectbox('Select X', ['day'], 0)


        fig = px.line(df, x=x, y=y, title='Emission Schedule')
        # add vertical lines for halving



        # add a lien for the total supply
        # ensure the scales are side by side

        st.plotly_chart(fig)

        fig = px.line(x=df['day'], y=df['total_emission'], title='Total Emission')

        st.plotly_chart(fig)





SubspaceDashboard.run(__name__)
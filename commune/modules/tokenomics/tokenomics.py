import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px
import streamlit as st


class SubspaceDashboard(c.Module):
    

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

        
            daily_dividends_per_token = current_emission_per_day * (1 / total_emission) * dividend_rate
            current_burned_emission_per_day = (current_emission_per_day * burn_rate) / n
            state.required_stake_to_cover_burn.append(current_burned_emission_per_day / daily_dividends_per_token)
            state.dividends_per_token.append(daily_dividends_per_token)


            state.burn_price_per_day.append(current_burned_emission_per_day)

            
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
    
    def dividends_dashboard(self, df):

        with st.expander('Dividends Calculator', expanded=True):

            my_stake = st.number_input('My Stake', 0, 1000000000000, 1000, 1, key=f'my.stake')
            final_stake = my_stake
            stake_over_time = [my_stake]
            appreciation = []


            for i in range(len(df['dividends_per_token'])):
                burn_price_per_day = df['burn_price_per_day'][i]
                stake_over_time.append(df['dividends_per_token'][i] * stake_over_time[-1] + stake_over_time[-1] - burn_price_per_day)
                if stake_over_time[-1] < 0:
                    stake_over_time[-1] = 0
                
                appreciation.append(stake_over_time[-1] / (stake_over_time[0] + 1e-10))
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

    @classmethod
    def dashboard(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        st.write('# Tokenomics')
        
        with st.expander('Parameters', expanded=False):
            cols = st.columns(2)

            emission_per_day = cols[0].number_input('Emission Per Day', 0, 1_000_000, 250_000, 1)
            
            starting_emission = cols[1].number_input('Starting Emission', 0, 100_000_000_000, 60_000_000, 1) 

            days = st.slider('Days', 1, 3_000, 800, 1)

            n = st.number_input('Number of Modules', 1, 1000000, 8400, 1, key=f'n.modules')

            dividend_rate = st.slider('Dividend Rate', 0.0, 1.0, 0.5, 0.01, key=f'dividend.rate')

            emission_per_halving = st.number_input('Emission Per Halving', 0, 1_000_000_000, 250_000_000, 1, key=f'emission.per.halving')


    
        emission_per_day = emission_per_day
        starting_emission = starting_emission
        days = days
        n = n
        burn_rate = burn_rate
        dividend_rate = dividend_rate
        burned_emission_per_day = emission_per_day * burn_rate
        block_time = 8
        tempo = 100
        seconds_per_day =  24 * 60 * 60
        seconds_per_year = 365 * seconds_per_day
        blocks_per_day = seconds_per_day / block_time
        blocks_per_year = seconds_per_year / block_time
        emission_per_block = emission_per_day / blocks_per_day
        emission_per_year = blocks_per_year * emission_per_block


        state = {
            'emission_per_day': emission_per_day,
            'starting_emission': starting_emission,
            'days': days,
            'n': n,
            'burn_rate': burn_rate,
            'dividend_rate': dividend_rate,
            'burned_emission_per_day': burned_emission_per_day ,
            'block_time': block_time,
            'tempo': tempo,
            'seconds_per_day':  seconds_per_day,
            'seconds_per_year': seconds_per_year,
            'blocks_per_day': blocks_per_day ,
            'blocks_per_year': blocks_per_year ,
            'emission_per_block': emission_per_block,
            'emission_per_year': emission_per_year ,
        }


        df = self.emission_schedule(days=days, 
                                    starting_emission=starting_emission, 
                                    emission_per_day=emission_per_day, 
                                    emission_per_halving=emission_per_halving,
                                    burn_rate = burn_rate,
                                    n=n,
                                    dividend_rate=dividend_rate


                                    )
        

        self.dividends_dashboard(df)

        st.markdown('## Emission Schedule')
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
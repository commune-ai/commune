import commune as c
import json
import numpy as np
import os
import streamlit as st
import plotly.express as px
import datetime


class App(c.Module):
    def __init__(self, model = 'model.openrouter', score_module='rvb'):
        self.model = c.module(model)()
        self.blue_model = c.module(score_module)()
        st.write('App Initialized', score_module)
    

    def signin(self):
        st.write('## Sign In')
        secret = st.text_input('whats your secret ;) ? ', 'sup', type='password')
        self.key = c.pwd2key(secret)
        st.write('My Public Address')
        st.code(self.key.ss58_address)
        return self.key
    
    def add_history(self, text):
        return self.put(f'history/{self.key.ss58_address}', text)
    
    def get_history(self, address=None, model=None):
        history_paths = self.get_history_paths(address=address, model=model)
        history = [self.get_json(fp) for fp in history_paths]
        return history
    

    def get_history_paths(self, address=None, model=None):
        address = address or self.key.ss58_address
        history_paths = []
        model_paths = [self.resolve_path(f'history/{model}')] if model else self.ls('history')
        for model_path in model_paths:
            user_folder = f'{model_path}/{address}'
            if not self.exists(user_folder):
                continue
            for fp in self.ls(user_folder):
                history_paths += [fp]
        return history_paths
    

    def global_history_paths(self):
        return self.glob('history/**')
    
    def global_history(self):
        history = []
        for path in self.global_history_paths():
            history += [self.get_json(path)]
        return history
    

    def clear_history(self):
        return [self.rm(path) for path in self.global_history_paths()]
    

    def derive_path(self, address, model):
        model = model.replace('/', '::')
        return f'history/{model}/{address}/{c.time()}.json'
    

    def model_arena(self):

        cols = st.columns([3,1])
        model = cols[0].selectbox('Select a model', self.blue_model.models())
        text = st.text_area('Enter your text here')
        for i in range(2):
            cols[1].write('\n')
        submit = cols[1].button('Attack the model')

        if submit:

            with st.status('Attacking the model'):
                red_response = self.model.forward(text, model=model)

            with st.expander('Red Model Response'):
                st.write(red_response)
            with st.status('Blue Model Response'):
                response = self.blue_model.score(red_response)# dict where mean is the score
            # plot mean in plotly in a pie chart between 0 and 1 where i can see it if it is 0
            # how do i have red for jailbreak and blue for not jailbreak
            plot = px.pie(values=[response['mean'], 1-response['mean']], names=['Jailbreak', 'Not Jailbreak'], title='Jailbreak Score')
            plot.update_traces(marker=dict(colors=['red', 'blue']))
            st.plotly_chart(plot)
        
            response['model'] = model
            response['address'] = self.key.ss58_address
            path = self.derive_path(address=self.key.ss58_address, model=model)
            self.put_json(path, response)

    def my_history(self, columns=['mean', 'timestamp', 'model', 'address'], sort_by='timestamp', ascending=False, model=None):
        df = c.df(self.get_history(model=model))
        if len(df) > 0:
            df = df[columns].sort_values(sort_by, ascending=ascending)
        else:
            st.write('No history found')
            return df
        # convert timestmap to human readable
        df['time'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        return df

    def stats(self, 
              columns=['mean', 'timestamp', 'model', 'address'],
              group_by = ['address', 'model'], 
              sort_by='mean', ascending=False, model=None):
        st.write('# Stats')
        cols = st.columns([4,1])
        for i in range(2):
            cols[0].write('\n')

        mode = st.selectbox('Mode', ['global', 'personal'])
        if mode == 'global':
            df = c.df(self.global_history())
        elif mode == 'personal':
            df = c.df(self.my_history())
        else:
            raise ValueError('Invalid mode')
        if len(df) == 0:
            return df
        
        
        # PROCESS THE DATA
        df = df[columns].sort_values(sort_by, ascending=ascending)
        # convert timestmap to human readable
        df['time'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        del df['timestamp']
        # select a model
        models = ['ALL'] + list(df['model'].unique())
        model = st.selectbox('Select a models', models, 0)
        group_by = st.multiselect('Group by', df.columns, group_by)
        if model != 'ALL':
            df = df[df['model'] == model]
        # group based on address
        if len(group_by) > 1:
            # add std and mean over the address with count of the number of scores
            st.write(df.groupby(group_by)['mean'].agg(['mean', 'count']).reset_index())
        else:
            df = df
            st.write(df)


        df = df.sort_values('mean', ascending=False)
        

        # truncate the address to 5 characters
        address_df = df.groupby('address')['mean'].agg(['mean']).reset_index()
        address_df = address_df.sort_values('mean', ascending=False)
        fig = px.bar(address_df, x='address', y='mean', title=f'Account Level Jailbreak Scores')
        st.plotly_chart(fig)

        model_df = df.groupby('model')['mean'].agg(['mean']).reset_index()
        model_df = model_df.sort_values('mean', ascending=False)
        fig = px.bar(model_df, x='model', y='mean', title=f'Model Level Jailbreak Scores')
        st.plotly_chart(fig)
        

    def app(self):
        with st.sidebar:
            st.write('# Always Blue')
            self.signin()

        fns = [ 'model_arena', 'stats']
        tabs = st.tabs(fns)
        for i, fn in enumerate(fns):
            with tabs[i]:
                getattr(self, fn)()


App.run(__name__)
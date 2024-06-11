import commune as c
import json
import numpy as np
import os
import streamlit as st
import plotly.express as px
import datetime

class App(c.Module):
    def __init__(self, model = 'model.openrouter', score_module=None):
        self.model = c.module(model)()
        if score_module != None:
            self.score_model = c.module(score_module)()
        else:
            self.score_model =  c.import_object('score_model.JailbreakScoreModel')()
        
    def signin(self):
        st.write('## Sign In')
        secret = st.text_input('What is your secret?', 'sup', type='password')
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
    




    def arena(self):

        cols = st.columns([3,1])
        model = cols[0].selectbox('Select a model', self.score_model.models())
        text = st.text_area('Red Team Prompt')
        for i in range(2):
            cols[1].write('\n')
        submit = cols[1].button('Submit Attack')
        cancel_attack = cols[1].button('Cancel Attack')
        attack_model = submit and not cancel_attack
        if attack_model:
            with st.status('Attacking the model', expanded=True):
                red_response = self.model.forward(text, model=model)
                st.write(red_response)
            with st.status('Blue Model Response', expanded=True):
                result = self.score_model.score(red_response)# dict where mean is the score
                result['model'] = model
                result['address'] = self.key.ss58_address
                self.save_result(result)
                # remove the legend
                
                plot = px.pie(values=[result['mean'], 1-result['mean']], names=['Jailbreak', 'Not Jailbreak'], title='Jailbreak Score')
                plot.update_traces(textinfo='percent+label', marker=dict(colors=['red', 'blue']))
                plot = plot.update_layout(showlegend=False)
                plot.update_traces(marker=dict(colors=['red', 'blue']))
                # have the plot fix the size of the plot
                plot.update_layout(autosize=False, width=500, height=500)
                st.plotly_chart(plot)

    def save_result(self, response):
        model = response['model']
        address = response['address']
        model = model.replace('/', '::')
        path =  f'history/{model}/{address}/{c.time()}.json'
        self.put_json(path, response)

    def my_history(self, columns=['mean', 'timestamp', 'model', 'address'], sort_by='timestamp', ascending=False, model=None):
        df = c.df(self.get_history(model=model))
        if len(df) > 0:
            df = df[columns].sort_values(sort_by, ascending=ascending)
        else:
            st.write('No history found')
            return df
        # convert timestmap to human readable
        df['time'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%m-%d %H:%M'))
        del df['timestamp']
        del df['address']
        df = df.set_index('time')
        df.sort_index(inplace=True)
        return df

    def stats(self, 
              columns=['mean', 'timestamp', 'model', 'address'],
              group_by = ['address', 'model'], 
              sort_by='mean', ascending=False, model=None):
        cols = st.columns([4,1])
        for i in range(2):
            cols[0].write('\n')

        df = c.df(self.global_history())
      
        if len(df) == 0:
            st.error('No history found, enter the arena')
            return df
        
        
        # PROCESS THE DATA
        df = df[columns].sort_values(sort_by, ascending=ascending)
        # convert timestmap to human readable
        df['time'] = df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
        del df['timestamp']
        # select a model
        models = ['ALL'] + list(df['model'].unique())
        model = st.selectbox('Select a models', models, 0)
        if model != 'ALL':
            df = df[df['model'] == model]
        # group based on address

        df = df
        with st.expander('Global'):
            st.write(df)

        with st.expander('Best Users'): 
            df = df.groupby('address')['mean'].agg(['mean']).reset_index()
            df = df.sort_values('mean', ascending=False)
            st.write(df)
    def sidebar(self, sidebar=True):
        if sidebar:
            with st.sidebar:
                return self.sidebar(sidebar=False)
    
        self.signin()









    def rules(self):
        st.write('## Description')
        st.write(' Increase the score of the red model by making it respond innapropriately')  


    def top_header(self):
        # have a random image
        st.write('# RedvBlue') 



    def app(self):
        self.sidebar()
        self.top_header()
        fns = [ 'arena', 'stats', 'rules']
        tabs = st.tabs(fns)
        for i, fn in enumerate(fns):
            with tabs[i]:
                getattr(self, fn)()


App.run(__name__)
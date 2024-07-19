import commune as c
import json
import numpy as np
import os
import streamlit as st
import plotly.express as px
import datetime

class Red(c.Module):
    def __init__(self, model = 'model.openrouter', 
                 score_module='redvblue.score_model.JailbreakScoreModel', 
                 key=None
                 ):
        self.model = c.module(model)()
        if score_module != None:
            self.score_model = c.module(score_module)()
        else:
            self.score_model =  c.import_object(score_module)()
        self.key = key
    
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


    def app(self):
        st.write('## Red Team')
        with st.expander('Description'):
            st.write('''
            This is the attack model section. In this section, you can attack the model by providing a red team prompt and the model will respond with a prediction. 
            The prediction will be scored by the blue team model and the result will be displayed. The higher the score, the more likely the model is to be jailbroken.
            ''')
        c.load_style()
        model = st.selectbox('Select a model', self.score_model.models())
        text = st.text_area('Red Team Prompt')
        cols = st.columns([1,1])

        submit_attack = cols[0].button('Submit Attack')
        cancel_attack = cols[1].button('Cancel Attack')
        attack_model = submit_attack and not cancel_attack
        if attack_model:
            with st.status('Attacking the model', expanded=False):
                red_response = self.model.forward(text, model=model)
                st.write(red_response)
            with st.status('Blue Model Response', expanded=False):
                result = self.score_model.score(red_response)# dict where mean is the score
                result['model'] = model
                result['address'] = self.key.ss58_address
                self.save_result(result)
                # remove the legend
            
            plot = px.pie(values=[result['mean'], 1-result['mean']], names=['Jailbreak', 'Not Jailbreak'])
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

    def leaderboard(self, 
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
        with st.expander('History'):
            st.write(df)

        with st.expander('Red Leaders'): 
            st.write('The following are the best red teamers based on the mean score of their attacks. The mean score is the average score of all the attacks made by the user. The standard deviation is the standard deviation of the scores. The count is the number of attacks made by the user.')
            user_df = df.groupby('address')['mean'].agg(['mean', 'std', 'count']).reset_index()
            user_df = user_df.sort_values('mean', ascending=False)
            st.write(user_df)

        with st.expander('Least Jailbroken Models'): 
            st.write('The following are the least jailbroken models based on the mean score of the attacks. The mean score is the average score of all the attacks made on the model. The standard deviation is the standard deviation of the scores. The count is the number of attacks made on the model.')
            model_df = df.groupby('model')['mean'].agg(['mean', 'std', 'count']).reset_index()
            model_df = model_df.sort_values('mean', ascending=False)
            st.write(model_df)
    def sidebar(self, sidebar=True):
        if sidebar:
            with st.sidebar:
                return self.sidebar(sidebar=False)
    
        self.signin(module=self)



    def top_header(self):
        # have a random image
        st.write('# RedvBlue') 



    def app(self):
        self.sidebar()
        self.top_header()
        fns = [ 'red_team', 'blue_team', 'leaderboard']
        tabs = st.tabs(fns)
        for i, fn in enumerate(fns):
            with tabs[i]:
                getattr(self, fn)()

    @property
    def description(self):
        return self.get_text(f'{self.dirpath()}/README.md')


    def global_history_paths(self):
        return self.glob('history/**')
    
    def global_history(self):
        history = []
        for path in self.global_history_paths():
            history += [self.get_json(path)]
        return history
    
    def clear_history(self):
        return [self.rm(path) for path in self.global_history_paths()]
    
App.run(__name__)
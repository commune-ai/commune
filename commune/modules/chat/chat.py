import commune as c
import streamlit as st
from .history import History



class Chat(c.Module):

    max_tokens = 100000

    def __init__(self, max_tokens=100000, history_path='history', **kwargs):
        self.max_tokens = max_tokens
        self.history = History(history_path=self.resolve_path(history_path))
        self.key = None

    def app(self):
        
        self.model = c.module('model.openrouter')()
        with st.sidebar:
            self.sidebar()
        self.chat_page()
        self.user_page()


    def models(self, max_age=10000):
        path = 'models'
        models = self.get(path, max_age=max_age)
        if models is None:
            models = self.model.models()
            self.put(path, models)
        
        return models

    def sidebar(self):
        st.write('# Chat')
        pwd = st.text_input('Enter Password', 'vibes', type='password')
        self.key = c.pwd2key(pwd)
        cols = st.columns([2, 10])
        cols[0].write('## Key')
        cols[1].code(self.key.ss58_address)

    def user_page(self):
        user_history = self.user_history()

    def get_history_path(self):
        return self.resolve_path('user_history')
                                
   
    def get_user_directory(self, key=None):
        key = (key or self.key)
        return self.get_history_path() + '/' +key.ss58_address
    
    def get_user_path(self):
        user_directory = self.get_user_directory()
        path = f'{user_directory}/{c.time()}.json'
        return path

    def user_history(self):
        path = self.get_user_directory()
        return self.ls(path)
    


    def generate(self, text, model, temperature=0.5, max_tokens=1000, data=None, ticket=None):
        data = data or {}
        
        params = {'model': model, 'temperature': temperature, 'max_tokens': max_tokens}
        data = {'input': text, 'output': '',  'params': params}
        r =  self.model.generate(input, model=model, stream=1)
        for token in r:
            data['output'] += token
            yield token
        if ticket != None:
            c.get_ticket_user(ticket)
        self.history.add_history(data)


    def chat_page(self):
        st.write('## Chat')
    
        path = self.get_user_path()
        model = st.selectbox('Model', self.models())
        text = st.text_area('Input', 'what is 2+2?', height=200)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.5)
        max_tokens = st.slider('Max Tokens', 1, self.max_tokens, self.max_tokens)
        send_button = st.button('Send') 
        save_content = st.checkbox('Save Content', True)
        if send_button:
            r = self.generate(text,**params, data=data)
            st.write_stream(r)


            
        with st.expander('Content'):
            self.put_json(path, data)


    def user_history(self):
        path = self.resolve_path(f'users/{self.key.ss58_address}')
        return self.ls(path)
            
            


Chat.run(__name__)
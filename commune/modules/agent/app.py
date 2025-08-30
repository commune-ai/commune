import commune as c
import streamlit as st


class App:

    def __init__(self, 
                 max_tokens=420000, 
                 password = None,
                 text = 'Hello whaduop fam',
                 system_prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.',
                 model = None,
                 history_path='history',
                **kwargs):

        self.key = c.key()
        self.max_tokens = max_tokens
        self.text = text
        self.set_module(model, 
                        password = password,
                        history_path=history_path, 
                        system_prompt=system_prompt,
                        **kwargs)
        
    def set_module(self,
                    model, 
                   history_path='history', 
                   password=None,
                   system_prompt = 'The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.',
                    **kwargs):
        self.system_prompt = system_prompt
        self.admin_key = c.pwd2key(password) if password else self.key
        self.model = c.module('agent')(model=model)
        self.models = self.model.models()
        self.history_path = c.get_path(history_path)
        return {'success':True, 'msg':'set_module passed'}
    
    def add_files(self, files):
        cwd = st.text_input('cwd', './')
        files = c.glob(cwd)
        files = st.multi_select(files, 'files')
        file_options  = [f.name for f in files]


    def call(self, 
            input = 'whats 2+2?' ,
            temperature= 0.5,
            max_tokens= 1000000,
            model= 'anthropic/claude-3.5-sonnet', 
            system_prompt= 'make this shit work',
            key = None,
            stream=True, 
            ):
        # key = self.resolve_key(key)
        data = c.locals2kwargs(locals())
        signature = self.key.ticket(c.hash(data))
        return signature

    def get_path(self, path):
        return '~/.commune/agent/app/' + path
    
    def sidebar(self, user='user', password='password', seperator='::'):
        with st.sidebar:
            st.title('Just Chat')
            # assert self.key.verify_ticket(ticket)
            with st.expander('LOGIN'): 
                cols = st.columns([1,1])
                user_name = cols[0].text_input('User', user)
                pwd = cols[1].text_input('Password', password, type='password')
                seed = c.hash(user_name + seperator + pwd)
                self.key = c.str2key(seed)
                self.data = c.dict2munch({  
                                'user': user_name, 
                                'path': self.get_path('history/'+ self.key.ss58_address ),
                                'history': self.history(self.key.ss58_address)
                                })
    
    def search_history(self):
        search = st.text_input('Search')
        # if the search is in any of the columns
        history = c.copy(self.data.history)

        history = [h for h in history if search in str(h)]
        df = c.df(history)
        st.write(df)

    @classmethod 
    def run(cls):
        self = cls()
        self.sidebar()
        tab_names = ['Chat', 'History']
        tabs = st.tabs(tab_names)
        with tabs[0]:
            self.chat_page()
        with tabs[1]:
            self.history_page()

    def chat_page(self):
        with st.sidebar.expander('PARAMS', expanded=True):
            model = st.selectbox('Model', self.models)
            temperature = st.number_input('Temperature', 0.0, 1.0, 0.5)
            if hasattr(self.model, 'get_model_info'):
                model_info = self.model.get_model_info(model)
                max_tokens = min(int(model_info['context_length']*0.9), self.max_tokens)
            else:
                model_info = {}
                max_tokens = self.max_tokens
            max_tokens = st.number_input('Max Tokens', 1, max_tokens, max_tokens)
            system_prompt = st.text_area('System Prompt',self.system_prompt, height=200)

        input  = st.text_area('Text',self.text, height=100)
        input = input + '\n' + system_prompt


        params = {
            'input': input,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        cols = st.columns([1,1])
        send_button = cols[0].button('Send', key='send', use_container_width=True) 
        stop_button = cols[1].button('Stop', key='stop', use_container_width=True)
        if send_button and not stop_button:
            r = self.model.forward(params['input'], 
                                    max_tokens=params['max_tokens'], 
                                    temperature=params['temperature'], 
                                    model=params['model'],
                                    stream=True)
            # dank emojis to give it that extra flair
            emojis = '✅🤖💻🔍🧠🔧⌨️'
            reverse_emojis = emojis[::-1]
            with st.spinner(f'{emojis} Generating {reverse_emojis}'):
                st.write_stream(r)

    def post_processing(self, data):
        lambda_string = st.text_area('fn(x={model_output})', 'x', height=100)
        prefix = 'lambda x: '
        lambda_string = prefix + lambda_string if not lambda_string.startswith(prefix) else lambda_string
        lambda_fn = eval(lambda_string)
        try:
            output = data['data']['output']
            output = lambda_fn(output)
        except Exception as e:
            st.error(e)

    def history_page(self):
        history = self.data.history
        if len(history) == 0:
            st.error('No History')
            return
        else:
            cols = history[0].keys()
            selected_columns = st.multiselect('Columns', cols, cols)
            df = c.df(history)[selected_columns]
            st.write(df)
    def user_files(self):
        return c.get(self.data['path'])



    def save_data(self, address, data):
        return c.put(self.history_path + '/' + address +'/data.json', data)
    
    def get_data(self, address):
        return c.get(self.history_path + '/' + address +'/data.json')

        
    def clear_history(self, address):
        return c.rm(self.history_path +  '/'+ address)
    
    def history_paths(self, address:str=None):
        paths = []
        if address == None:
            for user_address in self.user_addresses():
                 paths += self.history_paths(user_address)
        else:
            paths = c.ls(self.history_path + '/'+ address)
        return paths
    
    def save_data(self, data):
        path = self.history_path + '/'+ data['address'] + '/' + str(data['time']) + '.json'
        return c.put(path, data)
    
    def history(self, address:str=None, columns=['datetime', 
                                                 'input', 
                                                 'output', 
                                                 'system_prompt',
                                                 'model', 
                                                 'temperature',
                                                   'max_tokens'], df=False):
        paths = self.history_paths(address)
        history =  []
        for i, path in enumerate(paths):
            try:
                print(paths)
                h = c.get(path)
                h.update(h.pop('data'))
                h['datetime'] = c.time2datetime(h.pop('time'))
                h = {k:v for k,v in h.items() if k in columns}
                history.append(h)
            except Exception as e:
                print(e)
        # sort by time
    
        history = sorted(history, key=lambda x: x['datetime'], reverse=True)
        if df:
            history = c.df(history)
        return history
    
    def user_addresses(self, display_name=False):
        users = [u.split('/')[-1] for u in c.ls(self.history_path)]
        return users

if __name__ == '__main__':
    App().run()
import commune as c
import streamlit as st

class Chat(c.Module):
    def chat(self, text= 'hey', module='model', fn='generate', network='local',  kwargs=None, prefix_match=True) -> int:
        kwargs = kwargs or {}
        model = c.connect(module, network=network,prefix_match=prefix_match)
        return getattr(model, fn)(text, **kwargs)
    
    def check_history(self, history):
        if not isinstance(history, list):
            return False
        conds = []
        for message in history:
            conds += [
                isinstance(message, dict),
                'role' in message,
                'content' in message
            ]
        return all(conds)
    

    def tool_selector(self, module=None, fn=None, salt=None, key=None):
        module = module or self.module
        fn = fn or self.fn
        salt = salt or self.salt
        key = key or self.key
        if module == None:
            module = st.selectbox('Select a Module', c.modules(), 0, key=f'{salt}.module')
        if fn == None:
            fn = st.selectbox('Select a Function', c.module(module).fns(), 0, key=f'{salt}.fn')
        return module, fn, salt, key

    @classmethod
    def dashboard(cls, module:c.Module = None, key=None):
        self = cls()
        import streamlit as st
        import random
        import time

        fn = 'generate'
        default_text = 'hey'
        with st.sidebar:
            search = st.text_input('search', '', key='search')
            network = st.selectbox('Network', ['local', 'remote', 'subspace'], 0, key='network')
            if key == None:
                key = st.selectbox('Select a Key', c.keys(), 0, key='key')
                self.key = c.get_key(key)
            if search == '':
                search=None
            namespace = c.namespace(network=network)
            servers = list(namespace.keys())
            server_name = st.selectbox('Select a Server', servers, 0, key='server_name')

            update = st.button('Update Server Info')
        server_address = namespace[server_name]
        try:
            server = c.connect(server_name)
        except Exception as e:
            st.error(e)
            return
        info_path = f'servers/{server_name}/info'
        server_info = c.get(info_path, {})
        if not c.exists(info_path) or len(server_info) == 0 or update:

            server_info = server.info()
            c.put(info_path, server_info)


        if fn not in server_info['schema']:
            st.error(f'{fn} not in {server_name}')
            return
        

        fn = st.selectbox('Select a Function', list(server_info['schema'].keys()), 0, key='fn')

        default_kwargs = server_info['schema'][fn]['default']

        with st.expander('Parameters', expanded=False):

            with st.form(key='chat'):
                chat_path : str = f'chat/{server}/defaults'
                kwargs = self.get(chat_path, default={})
                kwargs.update(default_kwargs)
                # fn  = getattr(server, fn)
                kwargs = self.function2streamlit(fn=fn, fn_schema=server_info['schema'][fn], salt='chat')
                chat_button = st.form_submit_button('set parameters')
                if chat_button:
                    response = self.put(chat_path, kwargs)
                kwargs = self.get(chat_path, default=kwargs)

        clear_history = st.button("NOTHING HAPPENED ;)")

        key = self.key
        key_address = key.ss58_address
        user_history_path = f'users/{key_address}/user_history'
        # Initialize chat history
        if clear_history:
            self.put(user_history_path, [])
        history = self.get(user_history_path, [])
        if not self.check_history(history):
            st.error(f'history is not valid: {history}, resetting')
            history = []
            self.put(user_history_path, history)
        # Display chat messages from history on app rerun
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input(default_text):
            history.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message(self.server_name):
                kwargs = {k:v for i, (k,v) in enumerate(kwargs.items()) if i > 0}
                # if 'history' in kwargs:
                #     kwargs['history'] = history
                prompt = c.dict2str({'prompt': prompt, 'history': history, 'instruction': "respond as the assistant"})
                
                st.write(prompt)
                response = getattr(server, fn)(prompt, **kwargs)
                if isinstance(response, dict):
                    for k in ['response', 'text', 'content', 'message']:
                        if k in response:
                            response = response[k]
                            break

                st.write(response)
                            
                if isinstance(response, str):
                    history.append({"role": "assistant", "content": response})
                    self.put(user_history_path, history)

            # Add user message to chat history
        
        


Chat.run(__name__)
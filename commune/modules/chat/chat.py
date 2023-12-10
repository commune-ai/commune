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
    
    @classmethod
    def dashboard(cls):
        self = cls()
        import streamlit as st
        import random
        import time

        fn = 'generate'
        default_text = 'hey'
        search = st.text_input('Search', 'model', key='search')
        if search == '':
            search=None

        namespace = c.namespace(search=search)
        servers = list(namespace.keys())
        server_name = st.selectbox('Select a Server', servers, 0, key='server_name')

        try:
            server = c.connect(server_name)
        except Exception as e:
            st.error(e)
            return
        info_path = f'servers/{server_name}/info'
        server_info = c.get(info_path, {})

        if not c.exists(info_path):
            server_info = server.info()
            c.put(info_path, server_info)
        
        if fn not in server_info['schema']:
            st.error(f'{fn} not in {server_name}')
            return

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
                if 'history' in kwargs:
                    kwargs['history'] = history
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
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
        cols = st.columns(2)
        if module == None:
            module = cols[0].selectbox('Select a Module', c.modules(), 0, key=f'{salt}.module')
        if fn == None:
            fn = cols[1].selectbox('Select a Function', c.module(module).fns(), 0, key=f'{salt}.fn')
        return module, fn, salt, key

    @classmethod
    def app(cls, server:c.Module = None, key=None, network='local', salt=None, fn=None, kwargs=None, prefix_match=True):
        c.new_event_loop()

        self = cls()
        import streamlit as st
        import random
        import time

        fn = 'generate'
        default_text = 'hey'
        if network == None:
            network = st.selectbox('Network', ['local', 'remote', 'subspace'], 0, key='network')
        with st.sidebar:
            if key == None:
                key = st.selectbox('Select a Key', c.keys(), 0, key='key')
                self.key = c.get_key(key)
        key = self.key
        key_address = key.ss58_address
        st.write(type(c.namespace))
        namespace = c.namespace(network=network)
        servers = list(namespace.keys())
        cols = st.columns(2)
        server_name = cols[0].selectbox('Select a Server', servers, 0, key='server_name')

        server_address = namespace[server_name]
        try:
            server = c.connect(server_address)
        except Exception as e:
            st.error(e)
            return
        info_path = f'servers/{server_name}/info'
        server_info = c.get(info_path, {})
        if not c.exists(info_path) or len(server_info) == 0:
            server_info = server.info()
            c.put(info_path, server_info)

        if fn not in server_info['schema']:
            st.error(f'{fn} not in {server_name}')
            return
        fn = cols[1].selectbox('Select a Function', list(server_info['schema'].keys()), 0, key='fn')

        default_kwargs = server_info['schema'][fn]['default']

        with st.expander('Parameters', expanded=False):

            with st.form(key='chat'):
                chat_path : str = f'chat/{server}/defaults'
                kwargs = self.get(chat_path, default={})
                kwargs.update(default_kwargs)
                # fn  = getattr(server, fn)
                kwargs = c.function2streamlit(fn=fn, fn_schema=server_info['schema'][fn], salt='chat')
                chat_button = st.form_submit_button('set parameters')
                if chat_button:
                    response = self.put(chat_path, kwargs)
                kwargs = self.get(chat_path, default=kwargs)
        key = c.get_key(key)
        user_history_path = None
        with st.sidebar:
            
            convo_name = st.text_input('New Convo', 'default', key='convo_name')
            
            cols = st.columns(2)
            new_convo = cols[1].button(f"New Convo")
            rm_convo = cols[0].button(f"Remove Convo")
            user_history_path = f'user_history/{key_address}/{convo_name}'

            if new_convo:
                self.put(user_history_path, [])
            
            if rm_convo:
                self.rm(user_history_path)
            user_convos = [f.split('/')[-1].split('.')[0] for f in self.ls(f'user_history/{key_address}')]

            select_convo = st.selectbox('Select a Convo', user_convos, 0, key='select_convo')

            user_history_path = f'user_history/{key_address}/{select_convo}'
            st.write(f'**address** {convo_name}')
            user_history_path = f'user_history/{key_address}/{convo_name}'
            # Initialize chat history

            history = self.get(user_history_path, [])

            with st.expander('Chat History', expanded=False):
                st.write(history)
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
                
                response = getattr(server, fn)(prompt, **kwargs)
                if isinstance(response, dict):
                    for k in ['response', 'text', 'content', 'message']:
                        if k in response:
                            response = response[k]
                            break

   
                if isinstance(response, str):
                    history.append({"role": "assistant", "content": response})
                    # Display assistant message in chat message container
                    self.put(user_history_path, history)
            if isinstance(response, str):
                with st.chat_message("assistant"):
                    st.markdown(response)


            # Add user message to chat history
        

Chat.run(__name__)
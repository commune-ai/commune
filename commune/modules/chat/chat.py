import commune as c
import streamlit as st

class Chat(c.Module):
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)

    def chat(self, text= 'hey', module='model', fn='generate', network='local',  kwargs=None, prefix_match=True) -> int:
        kwargs = kwargs or {}
        model = c.connect(module, network=network,prefix_match=prefix_match)
        return getattr(model, fn)(text, **kwargs)
    
    
    @classmethod
    def dashboard(cls):
        self = cls()
        import streamlit as st
        import random
        import time

        fn = self.fn
        search = st.text_input('Search', '', key='search')
        namespace = c.namespace(search=search)
        server_info = self.server_info
        

        if fn not in server_info['schema']:
            st.error(f'{fn} not in {server_name}')

            return

        default_kwargs = server_info['schema'][fn]['default']

        with st.expander('Parameters', expanded=False):

            with st.form(key='chat'):
                chat_path : str = f'chat/{server}/defaults'
                kwargs = self.get(chat_path, default={})
                kwargs.update(default_kwargs)
                kwargs = self.function2streamlit(fn=fn, fn_schema=server_info['schema'][fn], salt='chat')
                chat_button = st.form_submit_button('set parameters')
                if chat_button:
                    response = self.put(chat_path, kwargs)
                kwargs = self.get(chat_path, default=kwargs)


        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []


        clear_history = st.button("NOTHING HAPPENED ;)")
        if clear_history:
            st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
 
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)


            with st.chat_message(self.server_name):
                kwargs = {k:v for i, (k,v) in enumerate(kwargs.items()) if i > 0}
                if 'history' in kwargs:
                    kwargs['history'] = st.session_state.messages
                response = getattr(server, fn)(prompt, **kwargs)
                if isinstance(response, dict):
                    for k in ['response', 'text', 'content', 'message']:
                        if k in response:
                            response = response[k]
                            break
                            
                if isinstance(response, str):
                    st.session_state.messages.append({"role": "assistant", "content": response})

                st.write(response)


            # Add user message to chat history
        


Chat.run(__name__)
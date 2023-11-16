import commune as c
import streamlit as st
import pandas as pd
from streamlit.components.v1 import components
import plotly.express as px


class Dashboard(c.Module):
    
    def __init__(self, netuid = 0, network = 'main', ): 
        self.set_config(locals())
    
        st.set_page_config(layout="wide")
        self.st = c.module('streamlit')()
        self.st.load_style()

        # THIS IS A SPECIAL FUNCTION
        self.load_state(update=False)


    def sidebar(self, sidebar:bool = True):

        if sidebar:
            with st.sidebar:
                return self.sidebar(sidebar=False)

        st.title(f'COMMUNE')


        self.select_key()
        self.select_network()


    def select_network(self, network=None):
        if network == None:
            network = self.network
        if not c.server_exists('module'):
                c.serve(wait_for_server=True)

        self.networks = c.networks()
        network2index = {n:i for i,n in enumerate(self.networks)}
        index = network2index['local']
        self.network = st.selectbox('Select a Network', self.networks, index=index, key='network.sidebar')
        namespace = c.namespace(network=self.network)
        with st.expander('Namespace', expanded=True):
            search = st.text_input('Search Namespace', '', key='search.namespace')
            df = pd.DataFrame(namespace.items(), columns=['key', 'value'])
            if search != '':
                df = df[df['key'].str.contains(search)]
            st.write(f'**{len(df)}** servers with **{search}** in it')

            df.set_index('key', inplace=True)
            st.dataframe(df, width=1000)
        sync = st.button(f'Sync {self.network} Network'.upper(), key='sync.network')
        self.servers = c.servers(network=self.network)
        if sync:
            c.sync()








    def get_module_stats(self, modules):
        df = pd.DataFrame(modules)
        del_keys = ['stake_from', 'stake_to', 'key']
        for k in del_keys:
            del df[k]
        return df

    # for k in ['emission', 'stake']:
    #     df[k] = df[k].apply(lambda x: c.round_decimals(self.subspace.format_amount(x, fmt='j'), 2))

    # df.sort_values('incentive', inplace=True, ascending=False)
    # df = df[:max_rows]
    # st.write(df)
    # st.dataframe(df, width=1000)
    # # BAR OF INCENTIVES
    # options = ['emission', 'incentive', 'dividends']
    # selected_keys = st.multiselect('Select Columns', options, options, key='stats')

    # for i, k in enumerate(selected_keys):
    #     cols = st.columns(2)

    #     fig = px.line(df, y=k, x= 'name', title=f'{k.upper()} Per Module')
    #     cols[0].plotly_chart(fig)
    #     fig = px.histogram(df, y=k, title=f'{k.upper()} Distribution')
    #     cols[1].plotly_chart(fig)


    @classmethod
    def dashboard(cls, key = None):
        import streamlit as st
        # plotly
        self = cls()
        self.sidebar()


    def select_key(self):
        keys = c.keys()
        key2index = {k:i for i,k in enumerate(keys)}
        self.key = st.selectbox('Select Key', keys, key2index['module'], key='key.sidebar')
        key_address = self.key.ss58_address
        st.write(f'**address** {self.key.__dict__}')
        st.write('\n\n\n'*2)
        st.code(key_address)
        return self.key

    @classmethod
    def dashboard(cls):
        self = cls()
        self.sidebar()




        
        tabs = st.tabs(['SERVE', 'WALLET', 'PLAYGROUND']) 
        chat = False
        with tabs[0]: 
            self.modules_dashboard()  
        with tabs[1]:
            self.subspace_dashboard()
        with tabs[2]:
            self.playground_dashboard()
        if chat:
            self.chat_dashboard()

    def playground_dashboard(self):

        server2index = {s:i for i,s in enumerate(self.servers)}
        default_servers = [self.servers[0]]
        cols = st.columns([1,1])
        self.server_name = cols[0].selectbox('Select Server',self.servers, 0, key=f'serve.module.playground')
        self.server = c.connect(self.server_name, network=self.network)
        self.server_info = self.server.info(schema=True, timeout=2)
        self.server_schema = self.server_info['schema']
        self.server_functions = list(self.server_schema.keys())
        self.server_address = self.server_info['address']

        self.fn = cols[1].selectbox('Select Function', self.server_functions, 0)

        self.fn_path = f'{self.server_name}/{self.fn}'
        st.write(f'**address** {self.server_address}')
        with st.expander(f'{self.fn_path} playground', expanded=True):

            kwargs = self.function2streamlit(fn=self.fn, fn_schema=self.server_schema[self.fn], salt='sidebar')

            st.write(kwargs)
            cols = st.columns([1,1])
            timeout = cols[0].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{self.fn_path}')
            call = st.button(f'Call {self.fn_path}')
            if call:
                try:
                    response = getattr(self.server, self.fn)(**kwargs, timeout=timeout)
                except Exception as e:
                    e = c.detailed_error(e)
                    response = {'success': False, 'message': e}
                st.write(response)
    
        
    def remote_dashboard(self):
        st.write('# Remote')




    def chat_dashboard(self):
        import streamlit as st
        import random
        import time



        fn = self.fn

        server_name = self.server_name
        server  = self.server
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
        

        


    def modules_dashboard(self):
        import pandas as pd

        modules = c.modules()
        self.st.line_seperator()
        module2index = {m:i for i,m in enumerate(modules)}
        module_name  = st.selectbox('Select a Module', modules, module2index['agent'], key=f'serve.module')



        module = c.module(module_name)
        # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
                    

        with st.expander('serve'):
            cols = st.columns([2,2,1])
            tag = cols[0].text_input('tag', 'replica', key=f'serve.tag.{module}')
            tag = None if tag == '' else tag

            n = cols[1].number_input('Number of Replicas', 1, 30, 1, 1, key=f'serve.n.{module}')
            
            [cols[2].write('\n\n\n') for _ in range(2)]
            register = cols[2].checkbox('Register', key=f'serve.register.{module}')
            if register:
                stake = cols[2].number_input('Stake', 0, 100000, 1000, 100, key=f'serve.stake.{module}')
            st.write(f'### {module_name.upper()} kwargs')
            with st.form(key=f'serve.{module}'):
                kwargs = self.function2streamlit(module=module, fn='__init__' )

                serve = st.form_submit_button('Serve')


                if serve:

                    if 'None' == tag:
                        tag = None
                    if 'tag' in kwargs:
                        kwargs['tag'] = tag
                    for i in range(n):
                        try:
                            if tag != None:
                                s_tag = f'{tag}.{i}'
                            else:
                                s_tag = str(i)
                            response = module.serve( kwargs = kwargs, tag=s_tag, network=self.network)
                        except Exception as e:
                            e = c.detailed_error(e)
                            response = {'success': False, 'message': e}
            
                        if response['success']:
                            st.write(response)
                        else:
                            st.error(response)

        with st.expander('Code', expanded=False):
            code = module.code()
            st.markdown(f"""
                        ```python
                        {code}
                        ```
                        """)

        cols = st.columns([2,2])
            
        with st.expander('Add Server'):
            address = st.text_input('Server Address', '')
            add_server = st.button('Add Server')
            if add_server:
                c.add_server(address)
        
        with st.expander('Remove Server'):
            server = st.selectbox('Module Name', self.servers, 0)
            rm_server = st.button('Remove Server')
            if rm_server:
                c.rm_server(server)


    def subspace_dashboard(self):
        return c.module('subspace.dashboard').dashboard(key=self.key)
    
    @classmethod
    def dash(cls, *args, **kwargs):
        c.print('FAM')
        return cls.st('dashboard')
    
Dashboard.run(__name__)


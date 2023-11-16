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


    def sidebar(self):
        with st.sidebar:

            if not c.server_exists('module'):
                    c.serve(wait_for_server=True)
            self.network_dashboard(sidebar=False)
            self.servers = c.servers(network=self.network)
            self.module_name = st.selectbox('Select Server', self.servers, 0)
            self.module = c.connect(self.module_name, network=self.network)

            module_info_path = f'module_info/{self.module_name}'
            module_info = self.get(module_info_path, default={})
            if module_info == {}:
                try:
                    module_info = self.module.info(schema=True)
                except Exception as e:
                    st.error(f'Module Not Found -> {self.module_name} {e}')
                    return
            self.module_info = module_info
            self.module_schema = self.module_info['schema']
            self.put(module_info_path, self.module_info)

            self.module_functions = self.module_info['functions']
            self.module_address = self.module_info['address']

            self.fn = st.selectbox('Select Function', self.module_functions, 0)

            self.fn_path = f'{self.module_name}/{self.fn}'
            st.write(f'**address** {self.module_address}')
            with st.expander(f'{self.fn_path} playground', expanded=True):

                kwargs = self.function2streamlit(fn=self.fn, fn_schema=self.module_schema[self.fn], salt='sidebar')
                cols = st.columns([1,1])
                timeout = cols[0].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{self.fn_path}')
                call = st.button(f'Call {self.fn_path}')
                if call:
                    try:
                        response = getattr(self.module, self.fn)(**kwargs, timeout=timeout)
                    except Exception as e:
                        e = c.detailed_error(e)
                        response = {'success': False, 'message': e}
                    st.write(response)

            self.key = c.module('key.dashboard').dashboard(state=self.state)
    
            sync = st.button(f'Sync {self.network} Network'.upper(), key='sync.network')
            if sync:
                c.update_network(self.network)
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
        st.title(f'COMMUNE')
        self.sidebar()
        
        tabs = st.tabs(['CHAT', 'MODULES', 'SUBSPACE']) 
        st.write(self.key)
        chat = False
        with tabs[0]:
            chat = True
        with tabs[1]: 
            self.modules_dashboard()  
        with tabs[2]:
            self.subspace_dashboard()
        if chat:
            self.chat_dashboard()

    def playground_dashboard(self):
        info = self.module_info
        network = self.network
        module_address = self.module_address
        st.write('Name: ', info['name'])
        schema = info['schema']
        buttons = {}
        for fn, fn_schema in schema.items():
            with st.expander(fn, expanded=False):
                with st.form(key=f'{fn}.form'):
                    kwargs = self.function2streamlit(fn=fn, fn_schema=fn_schema)
                    buttons[fn] = st.form_submit_button(fn)
                    if buttons[fn]:
                        kwargs['network'] = network

                        result = c.submit(c.call, args=[module_address, fn], timeout=10, kwargs=kwargs, return_future=False)[0]
                        st.write('Result', result)


        
    def remote_dashboard(self):
        st.write('# Remote')




    def chat_dashboard(self):
        import streamlit as st
        import random
        import time



        fn = self.fn

        module_name = self.module_name
        module = self.module
        module_info = self.module_info
        

        if fn not in module_info['schema']:
            st.error(f'{fn} not in {module_name}')

            return

        default_kwargs = module_info['schema'][fn]['default']

        with st.expander('Parameters', expanded=False):

            with st.form(key='chat'):
                chat_path : str = f'chat/{module}/defaults'
                kwargs = self.get(chat_path, default={})
                kwargs.update(default_kwargs)
                kwargs = self.function2streamlit(fn=fn, fn_schema=module_info['schema'][fn], salt='chat')
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


            with st.chat_message(self.module_name):
                kwargs = {k:v for i, (k,v) in enumerate(kwargs.items()) if i > 0}
                if 'history' in kwargs:
                    kwargs['history'] = st.session_state.messages
                response = getattr(module, fn)(prompt, **kwargs)
                if isinstance(response, dict):
                    for k in ['response', 'text', 'content', 'message']:
                        if k in response:
                            response = response[k]
                            break
                            
                if isinstance(response, str):
                    st.session_state.messages.append({"role": "assistant", "content": response})

                st.write(response)


            # Add user message to chat history
        
        
    def network_dashboard(self, sidebar=True): 

        if sidebar:
            with st.sidebar:
                self.network_dashboard(sidebar=False) 
        n = c.module('namespace')()
        self.networks = n.networks()
        network2index = {n:i for i,n in enumerate(self.networks)}
        index = network2index['local']

        cols = st.columns([1,1])
        self.network = cols[0].selectbox('Select a Network', self.networks, index=index, key='network.sidebar')



    def modules_dashboard(self):
        import pandas as pd

        modules = c.modules()
        self.st.line_seperator()
        module2index = {m:i for i,m in enumerate(modules)}
        module  = st.selectbox('Select A Module', modules, module2index['agent'], key=f'serve.module')


        module = c.module(module)
        # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
                    

        with st.form(key='serve'):
            
            kwargs = self.function2streamlit(module=module, fn='__init__' )

            cols = st.columns([1,1,2])
            tag = cols[0].text_input('tag', 'replica', key=f'serve.tag.{module}')
            tag = None if tag == '' else tag

            n = cols[1].number_input('Number of Replicas', 1, 30, 1, 1, key=f'serve.n.{module}')

            serve = cols[2].form_submit_button('Serve')

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


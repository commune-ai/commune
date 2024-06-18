
import commune as c
import streamlit as st
import pandas as pd
import json

class ServerApp(c.Module):
    default_model='model.openrouter'

    name2fn = {
                'Serve': 'serve_modules', 
                'Playground': 'playground', 
                'Network': 'network_app'
               }

    def process_config(self, config):
        with st.expander('Initialize Config', expanded=True):
            cols = st.columns(3)
            for i, (k,v) in enumerate(config.items()):
                with cols[i % len(cols)]:
                    v = st.text_input(f'{k}', str(v), key=f'{k}')
                    k_type = type(v)
                    type_str = str(k_type).split("'")[1]
                    if v != None:
                        if type_str == 'int':
                            v = int(v)
                        if type_str == 'float':
                            v = float(v)
                        if type_str == 'bool':
                            v = bool(v)
                        if type_str == 'list':
                            v = json.loads(v)
                        if type_str == 'dict':
                            v = json.loads(v)
                    config[k] = v

            return config
                
    def serve_modules(self):
    
        module2idx = {m:i for i,m in enumerate(self.modules)}
        cols = st.columns(3)
        select_module = cols[0].selectbox('Select a Module', self.modules, module2idx[self.default_model], key='select_module')
        timeout = cols[1].number_input('Timeout', 1, 60, 5, key='timeout')
        server_name = st.text_input('Server Name', self.default_model, key='server_name')
        n = cols[2].number_input('Number of Servers', 1, 10, 1, key='n_servers')
        module = c.module(select_module)
        config = module.config()
        config = self.process_config(config)


        cols = st.columns(2)
        serve_button = st.button('Deploy Server')
        
        if serve_button:
            spinners = []
            futures = []
            for i in range(n):
                server_name_i = f'{server_name}{i}' if '::' in server_name else f'{server_name}::{i}'
                future = c.submit(c.serve, kwargs=dict(module=server_name_i, kwargs=config), timeout=timeout)
                setattr(future, 'server_name', server_name_i)
                futures.append(future)
            for f in c.as_completed(futures, timeout=timeout):
                server_name = f.server_name
                result = f.result()
                checkmark_i = u'\u2713'
                msg = f'{checkmark_i} {server_name} Deployed {checkmark_i}' if not c.is_error(result) else f'{server_name} Failed \u2717'
   
                with st.expander(msg):
                    st.write(result)
               

    def update(self):
        self.modules = c.modules()
        self.servers = c.servers()
        self.module2index = {m:i for i,m in enumerate(self.modules)}


    def sidebar(self):
        with st.sidebar:
            network = st.selectbox('Select Network', ['local', 'remote', 'subspace'], 0, key=f'network')
            update = st.button('Update')
            self.set_network(network=network)

            with st.expander('Add Server'):
                address = st.text_input('Server Address', '')
                add_server = st.button('Add Server')
                if add_server:
                    c.add_server(address)
            
            with st.expander('Remove Server'):
                server = st.selectbox('Module Name', list(self.namespace.keys()), 0)
                rm_server = st.button('Remove Server')
                if rm_server:
                    c.rm_server(server)


    def playground(self):
        from commune.play.play import Play
        Play().app(namespace = c.namespace()) 
    

    def network_app(self):
        st.write('Local Network')
        namespace = c.namespace(network='local')
        st.write(namespace)
        run_epoch_button = st.button('Run Epoch')
        if run_epoch_button:
            results = c.run_epoch()
            df = pd.DataFrame(results)
            st.write(df)

    def app(self):

        self.sidebar()
        # self.options = st.multiselect('Select Options', options, ['serve', 'code', 'search', 'playground'], key=f'serve.options')
        names = list(self.name2fn.keys())
        names = st.multiselect('Select Function', names, names, key='selected_names')
        tabs = st.tabs(names)
        for i, name in enumerate(names):
            with tabs[i]:
                fn = self.name2fn[name]
                getattr(self, fn)()

    def set_network(self, network='local'):
        self.namespace = c.namespace(network=network) # NAME -> ADDRESS
        self.servers = list(self.namespace.keys()) # api that wraps over a module
        self.server_addreses = list(self.namespace.values())
        self.modules = c.modules()


ServerApp.run(__name__)



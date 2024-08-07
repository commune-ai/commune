
import commune as c
import streamlit as st
import pandas as pd
import json

class ServerApp(c.Module):
    default_model='model.openrouter'

    name2fn = {
                'Serve': 'serve_app', 
                'Network': 'network_app'
               }

    def process_params(self, params):
        from munch import Munch
        if isinstance(params, Munch):
            params = c.munch2dict(params)

        with st.expander('Initialize Config', expanded=True):
            cols = st.columns(3)
            for i, (k,v) in enumerate(params.items()):
                with cols[i % len(cols)]:
                    v = st.text_input(f'{k}', str(v), key=f'{k}2')
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
                    params[k] = v

            return params
                
    def serve_app(self):
    
        module2idx = {m:i for i,m in enumerate(self._modules)}
        cols = st.columns(3)
        select_module = cols[0].selectbox('Select a Module', self._modules, module2idx[self.default_model], key='select_module')
        server_name = st.text_input('Server Name', select_module, key='server_name')
        module = c.module(select_module)
        params = module.params()
        params = self.process_params(params)

        cols = st.columns(2)
        serve_button = st.button('Deploy Server')

        if serve_button:
            with st.spinner('Deploying Server'):
                result = c.serve(server_name, params=params)
                st.write(result)

    def update(self):
        self._modules = c.get_modules()
        self.servers = c.servers()
        self.module2index = {m:i for i,m in enumerate(self._modules)}


    def sidebar(self, sidebar=True):
        if sidebar:
            with st.sidebar:
                return self.sidebar(False)
        
        network = st.selectbox('Select Network', ['local', 'remote', 'subspace'], 0, key=f'network')
        update = st.button('Update Network')
        self.set_network(network=network, update=update)

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

        
        namespace = c.namespace(network='local')
        st.write(namespace)

    def network_app(self):
        st.write('Local Network')
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

    def set_network(self, network='local', update=False, max_age=1000):

        self.namespace = c.namespace(network=network, update=update, max_age=max_age) # NAME -> ADDRESS
        self.servers = list(self.namespace.keys()) # api that wraps over a module
        self.server_addreses = list(self.namespace.values())
        self._modules = c.get_modules()


ServerApp.run(__name__)



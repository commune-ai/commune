
import commune  c
import streamlit as st
import pandas as pd
import json

class ServerApp(c.Module):
    name2fn = {
                'Serve Module': 'serve_dashboard', 
                'Playground': 'playground', 
                'Local Network': 'local_network'
               }

    def serve_dashboard(self, default_model='model.openrouter'):
    
        module2idx = {m:i for i,m in enumerate(self.modules)}
        cols = st.columns(3)
        select_module = cols[0].selectbox('Select a Module', self.modules, module2idx[default_model], key='select_module')
        server_name = cols[1].text_input('Server Name', default_model, key='server_name')
        n = cols[2].number_input('Number of Servers', 1, 10, 1, key='n_servers')
        module = c.module(select_module)

        with st.expander('Parameters', expanded=False):
            config = module.config()
            cols = st.columns(3)
            for i, (k,v) in enumerate(config.items()):
                with cols[i % len(cols)]:
                    k_type = type(v)
                    config[k] = st.text_input(f'{k}', str(v), key=f'{k}')
                    type_str = str(k_type).split("'")[1]
                    if config[k] != None:
                        if type_str == 'int':
                            config[k] = int(config[k])
                        if type_str == 'float':
                            config[k] = float(config[k])
                        if type_str == 'bool':
                            config[k] = bool(config[k])
                        if type_str == 'list':
                            config[k] = json.loads(config[k])
                        if type_str == 'dict':
                            config[k] = json.loads(config[k])

        cols = st.columns(2)
        serve_button = st.button('Deploy Server')
        if serve_button:
            result = c.serve(server_name, kwargs=config)
            st.write(result)
            


    def update(self):
        self.modules = c.modules()
        self.servers = c.servers()
        self.module2index = {m:i for i,m in enumerate(self.modules)}


    def sidebar(self):
        with st.sidebar:
            self.network = st.selectbox('Select Network', ['local', 'remote', 'subspace'], 0, key=f'network')
            self.update = st.button('Update')

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


    def playground(self):
        from commune.play.play import Play
        Play().app(namespace = c.namespace()) 
    

    def local_network(self):
        st.write('Local Network')
        namespace = c.namespace(network='local')
        st.write(namespace)
        run_epoch_button = st.button('Run Epoch')
        if run_epoch_button:
            results = c.run_epoch()
            df = pd.DataFrame(results)
            st.write(df)

    def app(self):
        self.update()
        self.sidebar()
        self.namespace = c.namespace(network=self.network)
        # self.options = st.multiselect('Select Options', options, ['serve', 'code', 'search', 'playground'], key=f'serve.options')
        names = list(self.name2fn.keys())
        fns = list(self.name2fn.values())

        name = st.selectbox('Select Function', names, 0, key='selected_names')
        fn = self.name2fn[name]
        getattr(self, fn)()


ServerApp.run(__name__)



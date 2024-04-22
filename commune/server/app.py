
import commune as c
import streamlit as st
import pandas as pd
import streamlit as st

class ServerDashboard(c.Module):

    def history_dashboard(self):


        history_paths = self.history()
        history = []
        import os
        for h in history_paths:
            if len(h.split('/')) < 3:
                continue
            row =  {
                    'module': h.split('/')[-2],
                    **c.get(h, {})
                }
        
            row.update(row.pop('data', {}))
            history.append(row)
        
        df = pd.DataFrame(history)
        address2key = {v:k for k,v in self.namespace.items()}

        if len(df) == 0:
            st.write('No History')
            return
        modules = list(df['module'].unique())
        
        module = st.multiselect('Select Module', modules, modules)
        df = df[df['module'].isin(module)]
        columns = list(df.columns)
        with st.expander('Select Columns'):
            selected_columns = st.multiselect('Select Columns', columns, columns)
            df = df[selected_columns]
        
        st.write(df) 
        self.plot_dashboard(df=df, key='dam', select_columns=False)

    @classmethod
    def dashboard(cls, network = None, key= None):
        import pandas as pd
        self = cls()
        
        self.st = c.module('streamlit')
        modules = c.modules()
        self.servers = c.servers()
        self.st.line_seperator()
        module2index = {m:i for i,m in enumerate(modules)}

        with st.sidebar:
            self.network = st.selectbox('Select Network', ['local', 'remote', 'subspace'], 0, key=f'network')
            update= st.button('Update')


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

        module = st.selectbox('Select a Module', modules, 0, key='select')
        try:
            self.module = c.module(module)   
        except Exception as e:
            st.error(f'error loading ({module})')
            st.error(e)
            return 


        self.namespace = c.namespace(network=self.network, update=update)
        

        launcher_namespace = c.namespace(search='module::', namespace='remote')
        launcher_addresses = list(launcher_namespace.values())

        pages = ['serve', 'code', 'history', 'playground']
        # self.options = st.multiselect('Select Options', options, ['serve', 'code', 'search', 'playground'], key=f'serve.options')

        tabs = st.tabs(pages)

        with tabs[0]:
            self.serve_dashboard(module=self.module)
        with tabs[1]:
            self.code_dashboard()
        with tabs[2]:
            self.history_dashboard()

        # for i, page in enumerate(pages):
        #     with tabs[i]:
        #         getattr(self, f'{page}_dashboard')()

        module_name = self.module.path()
        # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')
    def playground_dashboard(self):
        c.module('playground').dashboard()

ServerDashboard.run(__name__)
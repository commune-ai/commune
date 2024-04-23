
import commune as c
import streamlit as st
import pandas as pd

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




    @classmethod
    def access_dashboard(cls):
        # self = cls(module="module",  base_rate=2)
        st.title('Access')

        
        modules = c.modules()
        module = st.selectbox('module', modules)
        update = st.button('update')
        if update:
            refresh = True
        self = cls(module=module)
        state = self.state


        self.st = c.module('streamlit')()
        self.st.load_style()

        fns = self.module.fns()
        whitelist_fns = state.get('whitelist', [])
        blacklist_fns = state.get('blacklist', [])

        with st.expander('Function Whitelist/Blacklist', True):
            whitelist_fns = [fn for fn in whitelist_fns if fn in fns]
            whitelist_fns = st.multiselect('whitelist', fns, whitelist_fns )
            blacklist_fns = [fn for fn in blacklist_fns if fn in fns]
            blacklist_fns = st.multiselect('blacklist', fns, blacklist_fns )


        with st.expander('Function Rate Limiting', True):
            fn =  st.selectbox('fn', whitelist_fns,0)
            cols = st.columns([1,1])
            fn_info = state['fn_info'].get(fn, {'stake2rate': self.config.stake2rate, 'max_rate': self.config.max_rate})
            fn_info['max_rate'] = cols[1].number_input('max_rate', 0.0, 1000.0, fn_info['max_rate'])
            fn_info['stake2rate'] = cols[0].number_input('stake2rate', 0.0, fn_info['max_rate'], min(fn_info['stake2rate'], fn_info['max_rate']))
            state['fn_info'][fn] = fn_info
            state['fn_info'][fn]['stake2rate'] = fn_info['stake2rate']
            state['fn_info'] = {fn: info for fn, info in state['fn_info'].items() if fn in whitelist_fns}

            fn_info_df = []
            for fn, info in state['fn_info'].items():
                info['fn'] = fn
                fn_info_df.append(info)

            if len(fn_info_df) > 0:
                fn_info_df = c.df(fn_info_df)
                fn_info_df.set_index('fn', inplace=True)

                st.dataframe(fn_info_df, use_container_width=True)
        state['whitelist'] = whitelist_fns
        state['blacklist'] = blacklist_fns

        with st.expander('state', False):
            st.write(state)
        if st.button('save'):
            self.put(self.state_path, state)

        # with st.expander("ADD BUTTON", True):
            

        # stake_per_call_per_minute = st.slider('stake_per_call', 0, 100, 10)
        # call_weight = st.slider('call_weight', 0, 100, 10)



ServerDashboard.run(__name__)



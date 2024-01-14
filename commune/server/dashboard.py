
import commune as c
import streamlit as st
import pandas as pd
import streamlit as st

class ServerDashboard(c.Module):
    @classmethod
    def dashboard(cls):
        import pandas as pd
        self = cls()
        self.st = c.module('streamlit')

        modules = c.modules()
        self.servers = c.servers()
        self.st.line_seperator()
        module2index = {m:i for i,m in enumerate(modules)}

        with st.sidebar:
            cols = st.columns([1,1])
            self.network = cols[0].selectbox('Select Network', ['local', 'remote', 'subspace'], 0, key=f'network')
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
        self.module  = st.selectbox('Select a Module', modules, module2index['agent'], key=f'serve.module')    
        self.module = c.module(self.module)    


        self.namespace = c.namespace(network=self.network, update=update)
        

        launcher_namespace = c.namespace(search='module::', namespace='remote')
        launcher_addresses = list(launcher_namespace.values())

        options = ['serve', 'code', 'search', 'playground']
        self.options = st.multiselect('Select Options', options, ['serve', 'code', 'search', 'playground'], key=f'serve.options')
        tabs = st.tabs(self.options)
        for option in self.options:
            with tabs[options.index(option)]:
                getattr(self, f'{option}_dashboard')()

        module_name = self.module.path()
        # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')


    def serve_dashboard(self):
        with st.expander('SERVE'):
            module_name = self.module.path()
            tag = st.text_input('tag', 'replica', key=f'serve.tag.{module_name}')
            tag = None if tag == '' else tag
            server_name = st.text_input('server_name', module_name + "::" + tag, key=f'serve_name.{module_name}')
            st.write(f'### {module_name.upper()} kwargs')
            n = 1
            with st.form(key=f'serve.{module_name}'):
                kwargs = self.function2streamlit(module=module_name, fn='__init__' )

                serve = st.form_submit_button('SERVE')


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
                            response = self.module.serve( kwargs = kwargs, server_name=server_name, network=self.network)
                        except Exception as e:
                            e = c.detailed_error(e)
                            response = {'success': False, 'message': e}
            
                        if response['success']:
                            st.write(response)
                        else:
                            st.error(response)

  

    def code_dashboard(self):
        with st.expander('CODE', expanded=False):
            code = self.module.code()
            code = self.code_editor(code)
            save_code = st.button('Save Code')
            if save_code:
                filepath = self.module.filepath()
                st.write(code )

                # c.put_text(filepath, code)
        
        with st.expander('README', expanded=False):
            
            markdown = self.module.readme()
            st.markdown(markdown)

        cols = st.columns([2,2])

    def search_dashboard(self):
        search = st.text_input('Search', '', key=f'search')
        namespace = {k:v for k,v in self.namespace.items() if search in k}
        df = pd.DataFrame(namespace.values(), index=namespace.keys(), columns=['address'])
        st.dataframe(df)

    def playground_dashboard(self):
        

        server2index = {s:i for i,s in enumerate(self.servers)}
        default_servers = [self.servers[0]]
        cols = st.columns([1,1])
        self.server_name = cols[0].selectbox('Select Server',self.servers, 0, key=f'serve.module.playground')
        self.server = c.connect(self.server_name, network=self.network)
        
        
        try:
            self.server_info = self.server.info(schema=True, timeout=2)
        except Exception as e:
            st.error(e)
            return

        self.server_schema = self.server_info['schema']
        self.server_functions = list(self.server_schema.keys())
        self.server_address = self.server_info['address']

        self.fn = cols[1].selectbox('Select Function', self.server_functions, 0)

        self.fn_path = f'{self.server_name}/{self.fn}'
        st.write(f'**address** {self.server_address}')
        with st.expander(f'{self.fn_path} playground', expanded=True):

            kwargs = self.function2streamlit(fn=self.fn, fn_schema=self.server_schema[self.fn], salt='sidebar')

            cols = st.columns([3,1])
            timeout = cols[1].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{self.fn_path}')
            cols[0].write('\n')
            cols[0].write('\n')
        call = st.button(f'Call {self.fn_path}')
        if call:
            success = False
            latency = 0
            try:
                t1 = c.time()
                response = getattr(self.server, self.fn)(**kwargs, timeout=timeout)
                t2 = c.time()
                latency = t2 - t1
                success = True
            except Exception as e:
                e = c.detailed_error(e)
                response = {'success': False, 'message': e}
            emoji = '✅' if success else '❌'
            latency = str(latency).split('.')[0] + '.'+str(latency).split('.')[1][:2]
            st.write(f'Reponse Status ({latency}s) : {emoji}')
            st.code(response)
    
    def code_editor(self, code):
        from code_editor import code_editor
        return code_editor(code)

       
ServerDashboard.run(__name__)



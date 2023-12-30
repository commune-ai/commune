
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
        self.st.line_seperator()
        module2index = {m:i for i,m in enumerate(modules)}
        with st.sidebar:
            cols = st.columns([1,1])
            module_name  = cols[0].selectbox('Select a Module', modules, module2index['agent'], key=f'serve.module')        

        launcher_namespace = c.namespace(search='module::', namespace='remote')
        launcher_addresses = list(launcher_namespace.values())
        

        module = c.module(module_name)
        # n = st.slider('replicas', 1, 10, 1, 1, key=f'n.{prefix}')

        with st.expander('serve'):
            cols = st.columns([2,2,1])
            tag = st.text_input('tag', 'replica', key=f'serve.tag.{module}')
            tag = None if tag == '' else tag
            server_name = st.text_input('server_name', module_name + "::" + tag, key=f'serve_name.{module}')

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
                            response = module.serve( kwargs = kwargs, server_name=server_name, network=self.network)
                        except Exception as e:
                            e = c.detailed_error(e)
                            response = {'success': False, 'message': e}
            
                        if response['success']:
                            st.write(response)
                        else:
                            st.error(response)

  

        with st.expander('Code', expanded=False):
            code = module.code()
            code = self.code_editor(code)
            save_code = st.button('Save Code')
            if save_code:
                filepath = module.filepath()

                c.put_text(filepath, code)
        
        with st.expander('readme', expanded=False):
            
            markdown = module.readme()
            st.markdown(markdown)

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
            call = cols[0].button(f'Call {self.fn_path}')
            if call:
                try:
                    response = getattr(self.server, self.fn)(**kwargs, timeout=timeout)
                except Exception as e:
                    e = c.detailed_error(e)
                    response = {'success': False, 'message': e}
                st.write(response)
    
    def code_editor(self, code):
        from code_editor import code_editor

        return code_editor(code)

       
ServerDashboard.run(__name__)



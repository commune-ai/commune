import commune as c
import streamlit as st
import pandas as pd
from typing import *

class Server(c.Module):
    server_mode = 'http'
    @classmethod
    def serve(cls, 
              module:Any = None ,
              tag:str=None,
              network = 'local',
              port :int = None, # name of the server if None, it will be the module name
              server_name:str=None, # name of the server if None, it will be the module name
              kwargs:dict = None,  # kwargs for the module
              refresh:bool = True, # refreshes the server's key
              tag_seperator:str='::',
              update:bool = False,
              max_workers:int = None,
              mode:str = "thread",
              public: bool = False,
              verbose:bool = False,
              ):

        module_path = module
        module_class = c.module(module)
        # this automatically adds 
        module = module_class(**kwargs)
        module.tag = tag
        module.server_name = server_name
        module.key = server_name
        address = c.get_address(server_name, network=network)

        if hasattr(module, 'info'):
            self.info = module.info()




        if address != None and ':' in address:
            port = address.split(':')[-1]   

        if c.server_exists(server_name, network=network): 
            if refresh:
                c.print(f'Stopping existing server {server_name}', color='yellow') 
            else:  
                return {'success':True, 'message':f'Server {server_name} already exists'}


        # RESOLVE THE WHITELIST AND BLACKLIST
        if hasattr(module, 'whitelist') :
            whitelist = module.whitelist
        if len(whitelist) == 0 and module != 'module':
            # do not include parent  functions in the whitelist when making this public as this can be a security risk
            whitelist = module.functions(include_parents=False)

        whitelist = list(set(whitelist + c.helper_functions))
        blacklist = module.blacklist if hasattr(module, 'blacklist') else []
        setattr(module, 'whitelist', whitelist)
        setattr(module, 'blacklist', blacklist)

        c.module(f'server.{server_mode}')(module=module, 
                                          name=server_name, 
                                          port=port, 
                                          network=network, 
                                          max_workers=max_workers, 
                                          mode=mode, 
                                          public=public)



        return {'success':True, 
                'address':  f'{c.default_ip}:{port}' ,
                  'name':server_name, 
                  'module':module_path}

    @classmethod
    def test(cls) -> dict:
        servers = c.servers()
        c.print(servers)
        tag = 'test'
        module_name = c.serve(module='module', tag=tag)['name']
        c.wait_for_server(module_name)
        assert module_name in c.servers()

        response = c.call(module_name)
        c.print(response)

        c.kill(module_name)
        assert module_name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}


    def serve_dashboard(self, expand=False, module=None):
        if expand:
            with st.expander('SERVE', expanded=True):
                return self.serve_dashboard(expand=False, module=module)

        if module == None:
            modules = c.modules()
            module = st.selectbox('Select a Module', modules, 0)
            try:
                module = c.module(module)
            except Exception as e:
                st.error(e)
        
        module_name = module.path()

        st.write(f'### SERVE {module_name.upper()}')
        n = 1
        emoji = '\N{Construction Sign}'
        cols = st.columns([1,1])
        tag = cols[0].text_input('tag', 'replica', key=f'serve.tag.{module_name}')
        tag = None if tag == '' else tag  
        with st.expander(f'{emoji}kwargs {emoji}', expanded=True):
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
                            response = self.module.serve( kwargs = kwargs, network=self.network)
                        except Exception as e:
                            e = c.detailed_error(e)
                            response = {'success': False, 'message': e}
            
                        if response['success']:
                            st.write(response)
                        else:
                            st.error(response)

    

    def code_dashboard(self):
        
        with st.expander('CODE', expanded=True):
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
        st.dataframe(df, use_container_width=True)

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

    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__',
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                           salt = None,
                            mode = 'pm2'):
        import streamlit as st
        
        key_prefix = f'{module}.{fn}'
        if salt != None:
            key_prefix = f'{key_prefix}.{salt}'
        if module == None:
            module = cls
            
        elif isinstance(module, str):
            module = c.module(module)
        extra_defaults = {} if extra_defaults is None else extra_defaults
        kwargs = {}

        if fn_schema == None:

            fn_schema = module.schema(defaults=True, include_parents=True)[fn]
            if fn == '__init__':
                config = module.config(to_munch=False)
                extra_defaults = config
            fn_schema['default'].pop('self', None)
            fn_schema['default'].pop('cls', None)
            fn_schema['default'].update(extra_defaults)
            fn_schema['default'].pop('config', None)
            fn_schema['default'].pop('kwargs', None)
            
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        
        if cols == None:
            cols = [1 for i in list(range(int(len(fn_schema['input'])**0.5)))]
        if len(cols) == 0:
            return kwargs
        cols = st.columns(cols)

        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
            if fn_key in skip_keys:
                continue
            if k in fn_schema['input']:
                k_type = fn_schema['input'][k]
                if 'Munch' in k_type or 'Dict' in k_type:
                    k_type = 'Dict'
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            

            col_idx = col_idx % (len(cols))
            if type(v) in [float, int] or c.is_number(v):
                kwargs[k] = cols[col_idx].number_input(fn_key, v, key=f'{key_prefix}.{k}')
            elif v in ['True', 'False']:
                kwargs[k] = cols[col_idx].checkbox(fn_key, v, key=f'{key_prefix}.{k}')
            else:
                kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}')
        
        kwargs = cls.process_kwargs(kwargs, fn_schema)  
        
             
        
        return kwargs


    @classmethod
    def save_serve_kwargs(cls,server_name:str,  kwargs:dict, network:str = 'local'):
        serve_kwargs = c.get(f'serve_kwargs/{network}', {})
        serve_kwargs[server_name] = kwargs
        c.put(f'serve_kwargs/{network}', serve_kwargs)
        return serve_kwargs
    
    @classmethod
    def load_serve_kwargs(cls, server_name:str, network:str = 'local'):
        serve_kwargs = c.get(f'serve_kwargs/{network}', {})
        return serve_kwargs.get(server_name, {})


    @classmethod
    def all_history(cls,server='module',  mode='http'):
        return c.module(f'server.{mode}').all_history()



    @classmethod
    def has_serve_kwargs(cls, server_name:str, network='local'):
        serve_kwargs = c.get(f'serve_kwargs/{network}', {})
        return server_name in serve_kwargs
    
    def history(self, server=None, mode='http', **kwargs):
        return c.module(f'server.{mode}').history(server, **kwargs)
    
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

    def bro(self):
        return 0

Server.run(__name__)
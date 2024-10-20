import commune as c

Router = c.module('router')
class RouterDashboard(Router):

    @classmethod
    def dashboard(cls):
        import streamlit as st
        st.write('ROUTER')
        self = cls()
        self.network = st.selectbox('network', ['local', 'remote', 'subspace', 'bittensor'])
        self.namespace = c.namespace(network=self.network)
        self.playground_dashboard(network=self.network)

    def playground_dashboard(self, network=None, server=None):
        # c.nest_asyncio()
        import streamlit as st

        if network == None:
            network = st.selectbox('network', ['local', 'remote'], 0, key='playground.net')
        else:
            network = network
            namespace = self.namespace
        servers = list(namespace.keys())
        if server == None:
            server_name = st.selectbox('Select Server',servers, 0, key=f'serve.module.playground')
            server = c.connect(server_name, network=network)
        server_info = server.info()
        server_schema = server_info['schema']
        server_functions = list(server_schema.keys())
        server_address = server_info['address']

        fn = st.selectbox('Select Function', server_functions, 0)

        fn_path = f'{self.server_name}/{fn}'
        st.write(f'**address** {server_address}')
        with st.expander(f'{fn_path} playground', expanded=True):

            kwargs = self.function2streamlit(fn=fn, fn_schema=server_schema[fn], salt='sidebar')

            cols = st.columns([3,1])
            timeout = cols[1].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{fn_path}')
            cols[0].write('\n')
            cols[0].write('\n')
        call = st.button(f'Call {fn_path}')
        if call:
            success = False
            latency = 0
            try:
                t1 = c.time()
                response = getattr(server, fn)(**kwargs, timeout=timeout)
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
    
    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__',
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                           salt = None):
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
            if type(v) in [float, int] or c.is_int(v):
                kwargs[k] = cols[col_idx].number_input(fn_key, v, key=f'{key_prefix}.{k}')
            elif v in ['True', 'False']:
                kwargs[k] = cols[col_idx].checkbox(fn_key, v, key=f'{key_prefix}.{k}')
            else:
                kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}')
        kwargs = cls.process_kwargs(kwargs, fn_schema)       
        
        return kwargs
    
    
RouterDashboard.run(__name__)
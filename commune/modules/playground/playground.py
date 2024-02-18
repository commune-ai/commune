import commune as c
import streamlit as st

class Playground(c.Module):

    @classmethod
    def dashboard(cls, module:c.Module = None,  network=None):
        
        if module == None:
            update = False
            if network == None:
                network = st.selectbox('network', ['local', 'remote', 'subspace'], 0, key='playground.net')
                update = st.button('Update', key='playground.update')
            else:
                network = network
            namespace = c.namespace(network=network, update=update)
            servers = list(namespace.keys())
            module = st.selectbox('Select Module', servers, 0, key='playground.module')
            server = c.connect(module, network=network)
        else:
            server = c.connect(module) 

        info_path = f'infos/{module}'
        if not c.exists(info_path) :
            server_info = server.info()
            c.put(info_path, server_info)
        server_info = c.get(info_path, {})
        server_name = server_info['name']
        server_schema = server_info['schema']
        server_address = server_info['address']
        server_functions = list(server_schema.keys())
        fn = st.selectbox('Select Function', server_functions, 0)

        fn_path = f'{server_name}/{fn}'


        
        st.write(f'**address** {server_address}')
        with st.expander(f'{fn_path} playground', expanded=True):
            kwargs = c.function2streamlit(fn=fn, fn_schema=server_schema[fn], salt='sidebar')
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
    
    

Playground.run(__name__)
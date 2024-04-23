import commune as c
import streamlit as st

class Play(c.Module):

    @classmethod
    def app(cls, module:c.Module = None,  network=None):
        self = cls()
        c.load_style() 
        modules = c.modules()
        module = st.selectbox('Module', modules)
        kwargs = self.module2streamlit(module)

    @classmethod
    def module2streamlit(cls, 
                           module = None,
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                           key_prefix = 'play',
                           salt = None,
                            mode = 'pm2', **kwargs):
        extra_defaults = extra_defaults or {}
        module_name = module
        module = c.module(module)
        schema = module.schema(defaults=True, include_parents=False)
        fn = st.selectbox('Function', list(schema.keys()))
        fn_schema = schema[fn]
        for k in ['self', 'cls', 'config', 'kwargs']:
            fn_schema['default'].pop(k, None)
        
        with st.expander('Schema'):
            st.write(fn_schema)
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        

        cols = int(max(len(fn_schema['input'])**0.5, 1))
        
        if fn_schema['type'] == 'self':
            if not c.server_exists(module_name):
                c.serve(module_name, mode=mode, wait_for_server=True)
            module = c.connect(module_name)

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


        cols = st.columns([3,1])
        timeout = cols[1].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{fn}')

        call = st.button(f'Call {fn}')

        if call:
            t1 = c.time()
            try:
                response = getattr(module, fn)(**kwargs, timeout=timeout)
                emoji = '✅'
            except Exception as e:
                emoji = '❌'
                response = c.detailed_error(e)
            latency = c.time() - t1
            latency = str(latency).split('.')[0] + '.'+str(latency).split('.')[1][:2]
            st.write(f'Reponse Status ({latency}s) : {emoji}')
            st.write(response)

        return kwargs
    
Play.run(__name__)
import commune as c
import streamlit as st
import pandas as pd

class Play(c.Module):
    skip_vars = ['kwargs', 'args']
    def app(self, namespace=None, key_prefix='play'):
        kwargs = {}
        st.write('# Play')
        namespace = namespace or c.namespace()
        servers = list(namespace.keys())
        modules = list(set([ server.split('::')[0] for server in servers]))
        module2serers = {module: [server for server in servers if module in server] for module in modules}
        cols = st.columns(3)
        module = cols[0].selectbox('Select Module', modules)
        servers = module2serers[module]
        n = len(servers)
        max_n = cols[1].number_input('Max Servers', 1, n, n)
        servers = servers[:max_n]
        servers = st.multiselect('Select Server', servers, servers)
        module_name = module
        try:
            module = c.module(module)
        except Exception as e:
            st.error(e)
            return  
        schema = module.schema(defaults=True, include_parents=True)
        whitelist = c.whitelist + module.whitelist if hasattr(module, 'whitelist') else []
        fn = st.selectbox('Function', whitelist)
        timeout = cols[2].number_input('Timeout', 1, 100, 10, 1, key=f'timeout.{fn}')
        fn_code = module.fn_code(fn)
        
        fn_schema = schema[fn] if fn in schema else None
        
        if fn_schema == None:
            st.error(f'Function {fn} not found in {module_name}')
        
        for fn_name in ['self', 'cls', 'config', 'kwargs']:
            fn_schema['input'].pop(fn_name, None)

        fns = list(fn_schema['input'].keys())
        cols = int(max(len(fns)**0.5, 1))
        
        with st.expander('Parameters', expanded=False):
            cols = st.columns(cols)

            for i, (var_name, var_info) in enumerate(fn_schema['input'].items()):
                # var_name ex: 'a'
                var_default = var_info['default']
                var_type = var_info['type']
                if var_type.startswith('typing'):
                    var_type = var_type.split('.')[-1]
                is_positional_or_keyword = bool(var_name in ['kwargs', 'args'] and var_default == 'NA')
                if is_positional_or_keyword or var_name in self.skip_vars:
                    continue
                is_var_optional = bool(var_default != 'NA')
                fn_key = f'**{var_name} ({var_type}){"(OPTIONAL)" if is_var_optional else "(REQUIRED)"}**'
                col_idx = i % (len(cols))
                col = cols[col_idx]
                if type(var_default) in [float, int] or c.is_int(var_default):
                    kwargs[var_name] = col.number_input(fn_key, var_default, key=f'{key_prefix}.{var_name}')
                elif var_default in ['True', 'False']:
                    kwargs[var_name] = col.checkbox(fn_key, var_default, key=f'{key_prefix}.{var_name}')
                else:
                    kwargs[var_name] = col.text_input(fn_key, var_default, key=f'{key_prefix}.{var_name}')
            kwargs = self.process_kwargs(kwargs, fn_schema)       

        with st.expander('Code', expanded=False):
            st.write('Code')
            st.code(fn_code)
        if fn_schema['docs'] != None:
            with st.expander('Documentation'):
                st.write('Documentation')
                st.write(fn_schema['docs'])

        call = st.button(f'Call {fn}')

        if call:
            future2server = {}
            server2info = {}
            for server in servers:
                server2info[server] = {'time': c.time(), 'server': server, 'latency': None}
                server_address = namespace[server]
                future = c.submit(c.call , args=[server_address+'/' + fn], kwargs={'params': kwargs}, timeout=timeout)
                future2server[future] = server

            for future in c.as_completed(future2server):
                server = future2server[future]
                server2info[server]['latency'] = c.round(c.time() - server2info[server]['time'], 3)
                latency = server2info[server]['latency']

                try:
                
                    response = future.result()
                    emoji = '✅'
                except Exception as e:
                    emoji = '❌'
                    response = c.detailed_error(e)

                with st.expander(f'{server} : {emoji} ({latency}s)'):
                    st.write(response)

            df = pd.DataFrame(server2info).T
            st.write(df)

        return kwargs
    
Play.run(__name__)
import commune as c

class Peer(c.Module):
    netork = 'peer'
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    

    def num_peers(self):
        return len(self.peers())
    n_peers = num_peers
    
    def peers(self, network='remote'):
        return c.servers(search='module', network='remote')
    



    def peer_dashboard(self):
        import streamlit as st
        import pandas as pd

        with st.sidebar:
            cols = st.columns(2)
            search = cols[0].text_input('Search', 'module')
            peer2info = self.peer2info()

            st.write(list(peer2info.values())[0])

        peer_info_df = []
        for peer, info in peer2info.items():
            memory_fields = ['available', 'total', 'used']
            row = {'peer': peer}
            for field in memory_fields:
                row['memory_'+field] = info.get('hardware', {}).get('memory', {}).get(field, None)

            # disk fields
            disk_fields = ['total', 'used', 'free']
            for field in disk_fields:
                row['disk_'+field] = info.get('hardware', {}).get('disk', {}).get(field, None)
            peer_info_df += [row]
            row['num_modules'] = len(info.get('namespace', {}))
        
        peer_info_df = pd.DataFrame(peer_info_df)
        namespace = c.namespace(search=search, network='remote')
        ip2host = self.ip2host()

        with st.expander('Peers', expanded=False):
            for peer, info in peer2info.items():
                cols = st.columns([1,4])
                peer = ip2host.get(peer, peer)
                cols[0].write('#### '+peer)

                timestamp = info.get('timestamp', None)
                lag = c.time() - timestamp if timestamp != None else None
                if lag != None:
                    lag = round(lag, 2)
                    st.write(f'{lag} seconds ago')
                cols[1].write(info.get('hardware', {}))
                cols[1].write(info.get('namespace', {}))

            if len(namespace) == 0:
                st.error(f'No peers found with search: {search}')
                return
            n = cols[1].slider('Number of servers', 1, len(namespace), len(namespace))
            module_names = list(namespace.keys())[:n]
            module_names = st.multiselect('Modules', module_names, module_names)
            namespace = {k:v for k,v in namespace.items() if k in module_names}
            module_addresses = list(namespace.values())
            module_names = list(namespace.keys())
        
        if len(module_names) == 0:
            st.error('No modules found')
            return
        
        cols = st.columns(3)
        module_name = cols[0].selectbox('Module', module_names, index=0)
        module_address = namespace[module_name]
        c.print(f'Connecting to {module_name} {module_address}')
        module = c.connect(module_address)
        cache = cols[2].checkbox('Cache', True)

        cache_path = f'module_info_cache/{module_address}'
        t1 = c.time()
        if cache:
            module_info = self.get_json(cache_path, {})
        else:
            module_info = {}

        if len(module_info) == 0:
            st.write('Getting module info')
            
            module_info = module.info()
            self.put_json(cache_path, module_info)
        fns = list(module_info['schema'].keys())
        fn_name = st.selectbox('Function', fns, index=0)
        fn = getattr(module, fn_name)
        with st.expander(fn_name, expanded=False):
            kwargs = self.function2streamlit(fn=fn_name, fn_schema=module_info['schema'][fn_name])
        timeout = cols[1].number_input('Timeout', 1, 100, 10, key='timeout_fn')
        run = st.button(f'Run {fn_name}')
        if run:
            future2module = {}
            for module_address in module_addresses:
                kwargs['fn'] = fn_name
                future = c.submit(c.call, args=[module_address], kwargs=kwargs, return_future=True)
                future2module[future] = module_address
            
            futures = list(future2module.keys())
            modules = list(future2module.values())
            for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):
                if not ('idx' in result and 'result' in result):
                    continue

                module_name = modules[result['idx']]
                result = result['result']
                
                with st.expander(f'{module_name}', expanded=False):

                    st.markdown(f'### {module_name}')
                    if c.is_error(result):
                        st.error(result)
                    else:
                        st.write(result)
        
        t2 = c.time()
        st.write(f'Info took {t2-t1} seconds')


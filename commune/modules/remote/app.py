import commune as c
import streamlit as st
from typing import *
import json

class App(c.Module):
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)
        self.remote = c.module('remote')()

    @classmethod
    def dashboard(cls, module: str = None, **kwargs):
        if module:
            cls = c.module(module)
        c.new_event_loop()
        import streamlit as st
        c.load_style()
        st.title('Remote Dashboard')
        self = cls()
        self.sidebar()
        mode = ['ssh', 'peer']
        mode = st.selectbox('Mode', mode, index=0)
        getattr(self, f'{mode}_dashboard')()



    def peer_dashboard(self):
        import streamlit as st
        import pandas as pd

        cols = st.columns(2)
        search = cols[0].text_input('Search', 'module')
        peer2info = self.remote.peer2info()

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
        ip2host = self.remote.ip2host()

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
                        result
                        pass
                    else:
                        st.write(result)
        
        t2 = c.time()
        st.write(f'Info took {t2-t1} seconds')



    def filter_hosts_dashboard(self, host_names: list = None):

        host_map = self.remote.hosts()
        host_names = list(host_map.keys())

        # get the search terms
        search_terms = self.remote.search_terms()
        for k, v  in search_terms.items():
            search_terms[k] = st.text_input(k, v)     
        self.remote.set_search_terms(search_terms)
        host_map = self.remote.filter_hosts(**search_terms)
        host_names = list(host_map.keys())
        n = len(host_names)

        

        with st.expander(f'Hosts (n={n})', expanded=False):
            host_names = st.multiselect('Host', host_names, host_names)
        self.host_map = {k:host_map[k] for k in host_names}
        self.host2ssh = self.remote.host2ssh(host_map=host_map)


    def manage_hosts_dashboard(self):

        with st.expander('Add Host', expanded=False):
            st.markdown('## Hosts')
            cols = st.columns(2)
            host = cols[0].text_input('Host',  '0.0.0.0')
            port = cols[1].number_input('Port', 22, 30000000000, 22)
            user = st.text_input('User', 'root')
            pwd = st.text_input('Password', type='password')
            add_host = st.button('Add Host')

            if add_host:
                self.remote.add_host(host=host, port=port, user=user, pwd=pwd)

        with st.expander('Remove Host', expanded=False):
            host_names = list(self.remote.hosts().keys())
            rm_host_name = st.selectbox('Host Name', host_names)
            rm_host = st.button('Remove Host')
            if rm_host:
                self.remote.rm_host(rm_host_name)

        
    
    def ssh_dashboard(self):
        host_map = self.host_map
        host_names = list(host_map.keys())

           
        # progress bar
        # add splace to cols[2] vertically
        

        with st.expander('params', False):
            cols = st.columns([4,4,2,2])
            cwd = cols[0].text_input('cwd', '/')
            timeout = cols[1].number_input('Timeout', 1, 100, 10)
            if cwd == '/':
                cwd = None
            for i in range(2):
                cols[2].write('\n')

            st.write('Print Formatting')
            expanded = True
            cols = st.columns([4,1])
            num_columns = cols[1].number_input('Num Columns', 1, 10, 2)
            fn_code = cols[0].text_input('Function', 'x')

        cols = st.columns([5,1])
        cmd = cols[0].text_input('Command', 'ls')
        [cols[1].write('') for i in range(2)]
        sudo = cols[1].checkbox('Sudo')

        if 'x' not in fn_code:
            fn_code = f'x'
        fn_code = f'lambda x: {fn_code}'
        fn_code = eval(fn_code)                               

        run_button = st.button('Run')
        host2future = {}
        
        if run_button:
            for host in host_names:
                future = c.submit(self.remote.ssh_cmd, args=[cmd], kwargs=dict(host=host, verbose=False, sudo=sudo, search=host_names, cwd=cwd), return_future=True, timeout=timeout)
                host2future[host] = future

            futures = list(host2future.values())
            hosts = list(host2future.keys())
            cols = st.columns(num_columns)
            failed_hosts = []
            col_idx = 0

            errors = []

            try:
                for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):

                    host = hosts[result['idx']]
                    if host == None:
                        continue

                    host2future.pop(host)
                    result = result['result']
                    is_error = c.is_error(result)
                    msg = result['error'] if is_error else result.strip()

                    # get the colkumne
                    col_idx = (col_idx) % len(cols)
                    col = cols[col_idx]
                    col_idx += 1


                    # if the column is full, add a new column
                    with col:
                        msg = fn_code(msg)
                        emoji =  c.emoji("cross") if is_error else c.emoji("check")
                        title = f'{emoji} :: {host} :: {emoji}'
                        if is_error:
                            failed_hosts += [host]
                            errors += [msg]
                            with st.expander(title, expanded=expanded):
                                st.error(msg)
                        else:
                            with st.expander(title, expanded=expanded):
                                st.code(msg)
                

            except Exception as e:
                pending_hosts = list(host2future.keys())
                st.error(c.detailed_error(e))
                st.error(f"Hosts {pending_hosts} timed out")
                failed_hosts += pending_hosts
            
            failed_hosts2ssh = {h:self.host2ssh[h] for h in failed_hosts}
            with st.expander('Failed Hosts', expanded=False):
                for host, ssh in failed_hosts2ssh.items():
                    st.write(host)
                    st.code(ssh)

    def sidebar(self, **kwargs):
        with st.sidebar:
            self.filter_hosts_dashboard()
            self.manage_hosts_dashboard()


App.run(__name__)

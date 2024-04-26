import commune as c
import streamlit as st
import toml
from typing import *
import json
import pandas as pd

class App(c.Module):
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)
        self.remote = c.module('remote')()
    @classmethod
    def app(cls, module: str = None, **kwargs):
        if module:
            cls = c.module(module)
        c.new_event_loop()
        c.load_style()
        self = cls()
        self.sidebar()
        modes = ['ssh', 'manage_hosts']
        tabs = st.tabs(modes)
        for i,t in enumerate(tabs):
            with t:
                getattr(self, f'{modes[i]}_dashboard')()

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

            cols = st.columns(2)
            user = cols[0].text_input('User', 'root')
            pwd = cols[1].text_input('Password', type='password')
            name = cols[0].text_input('Name', user+'@'+host)
            add_host = st.button('Add Host')

            if add_host:
                self.remote.add_host(host=host, port=port, user=user, pwd=pwd)

        with st.expander('Remove Host', expanded=False):
            host_names = list(self.remote.hosts().keys())
            rm_host_name = st.selectbox('Host to Remove', host_names)
            rm_host = st.button('Remove Host')
            if rm_host:
                self.remote.rm_host(rm_host_name)

        with st.expander('Rename Host', expanded=False):
            host_names = list(self.remote.hosts().keys())
            rm_host_name = st.selectbox('Host Name', host_names)
            new_host_name = st.text_input('New Host Name')
            rename_host = st.button('Rename Host')
            if rename_host:
                host = self.remote.hosts()[rm_host_name]
                self.remote.add_host(host)

                self.remote.rm_host(rm_host_name)

        self.host2ssh_search()

    def host2ssh_search(self, expander=True):
        if expander:
            with st.expander('Host2ssh', expanded=False):
                self.host2ssh_search(expander=False)
                return
        search =  st.text_input("Filter")
        host2ssh = self.host2ssh
        for host, ssh in host2ssh.items():
            if search != None and search not in host:
                continue
            st.write(host)
            st.code(ssh)
    
    def ssh_dashboard(self):

        self.ssh_params()

        host_map = self.host_map
        host_names = list(host_map.keys())
        num_columns = self.num_columns
        expanded = self.expanded
        cwd = self.cwd
        timeout = self.timeout

        cols = st.columns([5,1])
        cmd = cols[0].text_input('Command', 'ls')
        [cols[1].write('') for i in range(2)]
        sudo = cols[1].checkbox('Sudo')
        fn_code = self.fn_code
        if 'x' not in fn_code:
            fn_code = f'x'
        fn_code = f'lambda x: {fn_code}'
        fn_code = eval(fn_code)                               
        cols = st.columns(2)
        run_button = cols[0].button('Run')

        # make this a stop button red

        stop_button = cols[1].button('Stop')

        host2stats = self.get('host2stats', {})
        
        future2host = {}
        if run_button and not stop_button:
            for host in host_names:
                future = c.submit(self.remote.ssh_cmd, args=[cmd], kwargs=dict(host=host, verbose=False, sudo=sudo, search=host_names, cwd=cwd), return_future=True, timeout=timeout)
                future2host[future] = host
                host2stats[host] = host2stats.get(host, {'success': 0, 'error': 0 })

            cols = st.columns(num_columns)
            failed_hosts = []
            col_idx = 0
            errors = []
            futures = list(future2host.keys())

            try:
                for future in c.as_completed(futures, timeout=timeout):

                    if host == None:
                        continue

                    host = future2host.pop(future)
                    stats = host2stats.get(host, {'success': 0, 'error': 0})
                    result = future.result()
                    is_error = c.is_error(result)
                    if not is_error:        
                        msg = result if is_error else result.strip()

                        # get the colkumne
                        col_idx = (col_idx) % len(cols)
                        col = cols[col_idx]
                        col_idx += 1


                        stats = host2stats.get(host, {'success': 0, 'error': 0})



                        # if the column is full, add a new column
                        with col:
                            msg = fn_code(msg)
                            emoji =  c.emoji("cross") if is_error else c.emoji("check")
                            title = f'{emoji} :: {host} :: {emoji}'

                            stats['last_success'] = c.time()
                            stats['success'] += 1
                            with st.expander(f'Results {host}', expanded=expanded):
                                st.write(title)
                                st.code(msg)

                    host2stats[host] = stats
        
            except Exception as e:
                pending_hosts = list(future2host.values())
                st.error(c.detailed_error(e))
                st.error(f"Hosts {pending_hosts} timed out")
                failed_hosts += pending_hosts
                for host in pending_hosts:
                    stats = host2stats[host]
                    stats['error'] += 1
                    host2stats[host] = stats
                errors += [c.detailed_error(e)] * len(pending_hosts)
            
            self.put('host2stats', host2stats)

            with st.expander('Failed Hosts', expanded=False):
                selected_failed_hosts = st.multiselect('Failed Hosts', failed_hosts, failed_hosts)
                delete_failed = st.button('Delete Failed')
                if delete_failed:
                    for host in selected_failed_hosts:
                        self.remote.rm_host(host)

                for host, error in zip(failed_hosts, errors):
                    st.write(f'**{host}**')
                    st.code(error)


    def ssh_params(self):

        with st.expander('params', False):
            cols = st.columns([4,4,2,2])
            cwd = cols[0].text_input('cwd', '/')
            timeout = cols[1].number_input('Timeout', 1, 100, 10)
            if cwd == '/':
                cwd = None
            for i in range(2):
                cols[2].write('\n')

            st.write('Print Formatting')
            expanded = st.checkbox('Expanded', True)
            cols = st.columns([4,1])
            num_columns = cols[1].number_input('Num Columns', 1, 10, 2)
            fn_code = cols[0].text_input('Function', 'x')

            self.cwd = cwd
            self.timeout = timeout
            self.num_columns = num_columns
            self.fn_code = fn_code
            self.expanded = expanded



    def sidebar(self, sidebar=True, **kwargs):

        if sidebar:
            with st.sidebar:
                return self.sidebar(sidebar=False, **kwargs)

        st.title('Remote Dashboard')

        self.filter_hosts_dashboard()
  


App.run(__name__)




# def peer_dashboard(self):

#     cols = st.columns(2)
#     search = cols[0].text_input('Search', 'module')
#     peer2info = self.remote.peer2info()

#     peer_info_df = []
#     for peer, info in peer2info.items():
#         memory_fields = ['available', 'total', 'used']
#         row = {'peer': peer}
#         for field in memory_fields:
#             row['memory_'+field] = info.get('hardware', {}).get('memory', {}).get(field, None)
#         # disk fields
#         disk_fields = ['total', 'used', 'free']
#         for field in disk_fields:
#             row['disk_'+field] = info.get('hardware', {}).get('disk', {}).get(field, None)
#         peer_info_df += [row]
#         row['num_modules'] = len(info.get('namespace', {}))
    
#     peer_info_df = pd.DataFrame(peer_info_df)
#     namespace = c.namespace(search=search, network='remote')
#     ip2host = self.remote.ip2host()

#     with st.expander('Peers', expanded=False):
#         for peer, info in peer2info.items():
#             cols = st.columns([1,4])
#             peer = ip2host.get(peer, peer)
#             cols[0].write('#### '+peer)

#             timestamp = info.get('timestamp', None)
#             lag = c.time() - timestamp if timestamp != None else None
#             if lag != None:
#                 lag = round(lag, 2)
#                 st.write(f'{lag} seconds ago')
#             cols[1].write(info.get('hardware', {}))
#             cols[1].write(info.get('namespace', {}))

#         if len(namespace) == 0:
#             st.error(f'No peers found with search: {search}')
#             return
#         n = cols[1].slider('Number of servers', 1, len(namespace), len(namespace))
#         module_names = list(namespace.keys())[:n]
#         module_names = st.multiselect('Modules', module_names, module_names)
#         namespace = {k:v for k,v in namespace.items() if k in module_names}
#         module_addresses = list(namespace.values())
#         module_names = list(namespace.keys())
    
#     if len(module_names) == 0:
#         st.error('No modules found')
#         return
    
#     cols = st.columns(3)
#     module_name = cols[0].selectbox('Module', module_names, index=0)
#     module_address = namespace[module_name]
#     c.print(f'Connecting to {module_name} {module_address}')
#     module = c.connect(module_address)
#     cache = cols[2].checkbox('Cache', True)

#     cache_path = f'module_info_cache/{module_address}'
#     t1 = c.time()
#     if cache:
#         module_info = self.get_json(cache_path, {})
#     else:
#         module_info = {}

#     if len(module_info) == 0:
#         st.write('Getting module info')
        
#         module_info = module.info()
#         self.put_json(cache_path, module_info)
#     fns = list(module_info['schema'].keys())
#     fn_name = st.selectbox('Function', fns, index=0)
#     fn = getattr(module, fn_name)
#     with st.expander(fn_name, expanded=False):
#         kwargs = self.function2streamlit(fn=fn_name, fn_schema=module_info['schema'][fn_name])
#     timeout = cols[1].number_input('Timeout', 1, 100, 10, key='timeout_fn')
#     run = st.button(f'Run {fn_name}')
    
#     if run:
#         future2module = {}
#         for module_address in module_addresses:
#             kwargs['fn'] = fn_name
#             future = c.submit(c.call, args=[module_address], kwargs=kwargs, return_future=True)
#             future2module[future] = module_address
        
#         futures = list(future2module.keys())
#         modules = list(future2module.values())
#         for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):
#             if not ('idx' in result and 'result' in result):
#                 continue

#             module_name = modules[result['idx']]
#             result = result['result']
            
#             with st.expander(f'{module_name}', expanded=False):

#                 st.markdown(f'### {module_name}')
#                 if c.is_error(result):
#                     result
#                     pass
#                 else:
#                     st.write(result)
    
#     t2 = c.time()
#     st.write(f'Info took {t2-t1} seconds')


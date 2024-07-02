import commune as c
import streamlit as st
from typing import *
import json

class App(c.module('remote')):
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)
    
    @classmethod
    def app(cls, module: str = None, **kwargs):
        if module:
            cls = c.module(module)
        c.new_event_loop()
        # c.load_style()
        self = cls()
        modes = ['SSH', 'MANAGE_HOSTS']
        # make this spand the whole page
        tabs = st.tabs(modes)
        for i,t in enumerate(tabs):
            with t:
                getattr(self, f'{modes[i].lower()}')()

    def filter_hosts_dashboard(self, host_names: list = None, expanded: bool = True, **kwargs):

        host_map = self.hosts()
        host_names = list(host_map.keys())

        # get the search terms
        search_terms = self.search_terms()
        search_terms['include'] = st.text_input('search', search_terms.get('include', ''))     
        self.set_search_terms(search_terms)
        host_map = self.filter_hosts(**search_terms)
        host_names = list(host_map.keys())
        n = len(host_names)
        host_names = st.multiselect(f'Hosts(n={n})', host_names, host_names)
        self.host_map = {k:host_map[k] for k in host_names}
        self.host2ssh = self.host2ssh(host_map=host_map)

    def manage_hosts(self):

        with st.expander('host2ssh', expanded=1):
            self.host2ssh_search(expander=False)
  
        with st.expander('Add Host', expanded=False):
            st.markdown('## Hosts')
            cols = st.columns(3)
            user = cols[0].text_input('User', 'root')
            host = cols[1].text_input('Host',  '0.0.0.0')
            port = cols[2].number_input('Port', 22, 30000000000, 22)

            cols = st.columns(2)
            pwd = cols[1].text_input('Password', type='password')
            name = cols[0].text_input('Name', user+'@'+host)
            add_host = st.button('Add Host')

            if add_host:
                r = self.add_host(host=host, port=port, user=user, pwd=pwd, name=name)
                st.write(r)
        with st.expander('Remove Host', expanded=False):
            host_names = list(self.hosts().keys())
            rm_host_name = st.selectbox('Host to Remove', host_names)
            rm_host = st.button('Remove Host')
            if rm_host:
                st.write(self.rm_host(rm_host_name))

        with st.expander('Rename Host', expanded=False):
            host_names = list(self.hosts().keys())
            rm_host_name = st.selectbox('Host Name', host_names)
            new_host_name = st.text_input('New Host Name')
            rename_host = st.button('Rename Host')
            if rename_host:
                host = self.hosts()[rm_host_name]
                self.add_host(host)
                self.rm_host(rm_host_name)


    def host2ssh_search(self, expander=True):
        host =  st.selectbox('Search', list(self.host2ssh.keys()))
        host2ssh = self.host2ssh
        host2ssh = host2ssh.get(host, {})
        st.code(host2ssh)
    
    def ssh(self):


        with st.expander('params', False):
            cols = st.columns([4,4,2,2])
            cwd = cols[0].text_input('cwd', '/')
            timeout = cols[1].number_input('Timeout', 1, 100, 10)
            if cwd == '/':
                cwd = None
            for i in range(2):
                cols[i].write('')
            self.sudo = cols[2].checkbox('Sudo')
            st.write('---')
            st.write('## Docker')
            cols = st.columns([2,1])
            enable_docker = cols[1].checkbox('Enable Docker')
            docker_container = cols[0].text_input('Docker Container', 'commune')

            # line 
            st.write('---')
            st.write('## Function')
            cols = st.columns([4,1])
            num_columns = cols[1].number_input('Num Columns', 1, 10, 2)
            fn_code = cols[0].text_input('Function', 'x')

            cwd = cwd
            timeout = timeout
            num_columns = num_columns
            fn_code = fn_code
            expanded = 1


        self.filter_hosts_dashboard()


        host_map = self.host_map
        cols = st.columns([5,1])
        
        cmd = cols[0].text_input('Command', 'ls')
        [cols[1].write('') for i in range(2)]
        if 'x' not in fn_code:
            fn_code = f'x'
        fn_code = f'lambda x: {fn_code}'
        fn_code = eval(fn_code)                               
        cols = st.columns(2)
        run_button = cols[0].button('Run')
        stop_button = cols[1].button('Stop')


        host2stats = self.get('host2stats', {})
        future2host = {}
        host_names = list(host_map.keys())
        if run_button and not stop_button:
            if enable_docker:
                cmd = f'docker exec {docker_container} {cmd}'
            for host in host_names:
                cmd_kwargs = dict(host=host, verbose=False, sudo=self.sudo, search=host_names, cwd=cwd)
                future = c.submit(self.ssh_cmd, args=[cmd], kwargs=cmd_kwargs, timeout=timeout)
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
        
            with st.expander('Failed Hosts', expanded=False):
                selected_failed_hosts = st.multiselect('Failed Hosts', failed_hosts, failed_hosts)
                delete_failed = st.button('Delete Failed')
                if delete_failed:
                    for host in selected_failed_hosts:
                        st.write(self.rm_host(host))

                for host, error in zip(failed_hosts, errors):
                    st.write(f'**{host}**')
                    st.code(error)




App.run(__name__)


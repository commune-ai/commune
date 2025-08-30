import commune as c
import streamlit as st
from typing import *
import json 
Remote = c.module('remote')
class App:
    def __init__(self, **kwargs):
        remote = Remote()
        for fn in dir(remote):
            if not fn.startswith('_'):
                try:
                    setattr(self, fn, getattr(remote, fn))
                except Exception as e:
                    print(e)
    @classmethod
    def app(cls, module: str = None, **kwargs):
        if module:
            cls = c.module(module)

        self = cls()
        with st.sidebar:

            self.filter_hosts_dashboard()
        tabs = st.tabs(['SSH', 'Manage Hosts'])

        with tabs[0]:
            self.ssh()
        with tabs[1]:
            self.manage_hosts()


    def edit_hosts(self):
        host_map = self._host_map
        og_host_map = self.hosts()
        og_host_map = {k:v for k,v in og_host_map.items() if k in host_map }
        # edit the hosts by edihting the string of the json
        st.write('## Hosts')    
        host_map = st.text_area('Host Map', json.dumps(host_map, indent=4), height=1000)
    
        save_hosts = st.button('Save Hosts')

        st.write(host_map)
        try: 
            host_map = json.loads(host_map)
        except Exception as e:
            st.error(e)
        host_map = host_map.update(og_host_map)
        if save_hosts:
            self.save_hosts(host_map)

        

    def filter_hosts_dashboard(self, host_names: list = None, expanded: bool = True, **kwargs):

        host_map = self.hosts()
        host_names = list(host_map.keys())

        search_terms = self.search_terms()
        self.set_search_terms(search_terms)

        search_terms['include'] = st.text_input('search', search_terms.get('include', ''))     
        self.set_search_terms(search_terms)
        host_map = self.filter_hosts(**search_terms)
        host_names = list(host_map.keys())
        n = len(host_names)
        
        # get the search terms
        with st.expander(f'Hosts(n={n})', expanded=True):   
            host_names = st.multiselect(f'Hosts', host_names, host_names)
            self._host_map = {k:host_map[k] for k in host_names}
            self.host2ssh = self.host2ssh(host_map=host_map)

    def manage_hosts(self):

        with st.expander('host2ssh', expanded=1):
            self.host2ssh_search(expander=False)
        

        with st.expander('Add Host', expanded=False):
            st.markdown('## Hosts')
            host_map = self.hosts()
            default_host = st.selectbox('Copy Host', list(host_map.keys()))
            default_parmas = host_map[default_host]
            cols = st.columns(3)
            user = cols[0].text_input('User', default_parmas['user'])
            host = cols[1].text_input('Host',  default_parmas['host'])
            port = cols[2].number_input('Port', 22, 30000000000, default_parmas['port'])

            cols = st.columns(2)
            pwd = cols[1].text_input('Password',default_parmas['pwd'], type='password')
            name = cols[0].text_input('Name', default_host)
            metadata = st.text_area('Metadata', default_parmas.get('metadata', ''))
            add_host = st.button('Add Host')

            if add_host:
                r = self.add_host(host=host, 
                                  port=port, 
                                  user=user, 
                                  pwd=pwd, 
                                  name=name, 
                                  metadata=metadata)
                st.success(r)


   
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

        with st.expander('Edit Hosts', expanded=False):
            self.edit_hosts()

        with st.expander('Save SSH Config', expanded=True):
            path = st.text_input('Enter Path', "~/.ssh/config")
            if  st.button(f'Save SSH Config to {path}'):
                st.success(self.save_ssh_config(path))



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
            cols = st.columns([2,1])
            enable_docker = cols[1].checkbox('Enable Docker')
            docker_container = cols[0].text_input('Docker Container', 'commune')

            # line 
            st.write('---')
            cols = st.columns([4,1])
            num_columns = cols[1].number_input('Num Columns', 1, 10, 2)
            fn_code = cols[0].text_input('Function', 'x')

            cwd = cwd
            timeout = timeout
            num_columns = num_columns
            fn_code = fn_code
            expanded = 1



        host_map = self._host_map
        
        cmd = st.text_input('Command', 'ls')
        if 'x' not in fn_code:
            fn_code = f'x'
        fn_code = f'lambda x: {fn_code}'
        fn_code = eval(fn_code)  
        cols = st.columns([1,1])                             
        run_button = cols[0].button('Run', use_container_width=True)
        stop_button = cols[1].button('Stop', use_container_width=True)


        host2stats = c.get('host2stats', {})
        future2host = {}
        host_names = list(host_map.keys())
        if run_button and not stop_button:
            if enable_docker:
                cmd = f'docker exec {docker_container} {cmd}'
            for host in host_names:
                cmd_kwargs = dict(host=host, verbose=False, sudo=self.sudo, cwd=cwd)
                future = c.submit(self.ssh_cmd, args=[cmd], kwargs=cmd_kwargs, timeout=timeout)
                future2host[future] = host
                host2stats[host] = host2stats.get(host, {'success': 0, 'error': 0 })

            cols = st.columns(num_columns)
            failed_hosts = []
            errors = []
            futures = list(future2host.keys())
            cols = st.columns(num_columns)
            col_idx = 0
            try:
                for future in c.as_completed(futures, timeout=timeout):
                    if host == None:
                        continue
                    host = future2host.pop(future)
                    stats = host2stats.get(host, {'success': 0, 'error': 0})
                    result = future.result()
                    is_error = c.is_error(result)
                    emoji =  c.emoji("cross") if is_error else c.emoji("check")
                    stats = host2stats.get(host, {'success': 0, 'error': 0})
                    title = f'{emoji} :: {host} :: {emoji}'
                    st.write(result)
                    if not is_error:        
                        msg =  result.strip()
                        msg = fn_code(msg)
                        stats['last_success'] = c.time()
                        stats['success'] += 1
                        col = cols[col_idx % num_columns]
                        col_idx += 1
                        emoji = c.emoji("check")
                        with col.expander(f'{title}', expanded=0):
                            st.write(title)
                            st.code('\n'.join(msg.split('\n')))
                    else:
                        msg = result
                        stats['error'] += 1
                        failed_hosts.append(host)
                        errors.append(result)
 
        
                    
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

if __name__ == '__main__':
    App.app()




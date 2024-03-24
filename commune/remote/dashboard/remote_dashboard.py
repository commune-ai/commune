import commune as c
import streamlit as st

Remote = c.module('remote')

class RemoteDashboard(Remote):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())


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
        self.ssh_dashboard()

    def sidebar(self, **kwargs):
        with st.sidebar:
            self.filter_hosts()
            self.manage_hosts()
            with st.expander('host2ssh'):
                st.write(self.host2ssh)


    def filter_hosts(self, **kwargs):

        host_map = self.hosts()
        host_names = list(host_map.keys())

        search_terms_dict = {
            'include': '',
            'avoid': ''
        }

        for search_type, search_terms in search_terms_dict.items():
            search_terms = st.text_input(search_type, search_terms)
            if len(search_terms) > 0:
                if ',' in search_terms:
                    search_terms = search_terms.split(',')
                else:
                    search_terms = [search_terms]
                
                search_terms = [a.strip() for a in search_terms]
            else:
                search_terms = []

            search_terms_dict[search_type] = search_terms
        
        with st.expander('Search Terms', expanded=False):
            st.write( search_terms_dict)

        def filter_host(host_name):
            for avoid_term in search_terms_dict["avoid"]:
                if avoid_term in host_name:
                    return False
            for include_term in search_terms_dict["include"]:
                if not include_term in host_name:
                    return False
            return True

        host_map = {k:v for k,v in host_map.items() if filter_host(k)}
        host_names = list(host_map.keys())

        n = len(host_names)
        host2ssh =  {}
        with st.expander(f'Hosts (n={n})', expanded=False):
            host_names = st.multiselect('Host', host_names, host_names)
            for host_name, host in host_map.items():
                cols = st.columns([1,4])
                cols[0].write('#### '+host_name)
                host2ssh[host_name] = f'sshpass -p {host["pwd"]} ssh {host["user"]}@{host["host"]} -p {host["port"]}'
                st.code(host2ssh[host_name])
            self.host_map = host_map

        self.host2ssh = host2ssh
        self.host_map = host_map




    def manage_hosts(self):

        with st.expander('Add Host', expanded=False):
            st.markdown('## Hosts')
            cols = st.columns(2)
            host = cols[0].text_input('Host',  '0.0.0.0')
            port = cols[1].number_input('Port', 22, 30000000000, 22)
            user = st.text_input('User', 'root')
            pwd = st.text_input('Password', type='password')
            add_host = st.button('Add Host')

            if add_host:
                self.add_host(host=host, port=port, user=user, pwd=pwd)

        with st.expander('Remove Host', expanded=False):
            host_names = list(self.hosts().keys())
            rm_host_name = st.selectbox('Host Name', host_names)
            rm_host = st.button('Remove Host')
            if rm_host:
                self.rm_host(rm_host_name)

        
    
    def ssh_dashboard(self):
        import streamlit as st
        host_map = self.host_map
        host_names = list(host_map.keys())
    
        # add splace to cols[2] vertically
        
        with st.expander('params', False):
            cols = st.columns([4,4,2])
            cwd = cols[0].text_input('cwd', '/')
            timeout = cols[1].number_input('Timeout', 1, 100, 10)
            if cwd == '/':
                cwd = None
            fn_code = st.text_input('Function', '''x''')
            for i in range(2):
                cols[2].write('\n')
            filter_bool = cols[2].checkbox('Filter', False)

        cols = st.columns([5,1])
        cmd = cols[0].text_input('Command', 'ls')
        [cols[1].write('') for i in range(2)]
        sudo = cols[1].checkbox('Sudo')

        if 'x' not in fn_code:
            fn_code = f'x'

        fn_code = eval(f'lambda x: {fn_code}')                               

        run_button = st.button('Run')
        host2future = {}
        
        if run_button:
            for host in host_names:
                future = c.submit(self.ssh_cmd, args=[cmd], kwargs=dict(host=host, verbose=False, sudo=sudo, search=host_names, cwd=cwd), return_future=True, timeout=timeout)
                host2future[host] = future

            futures = list(host2future.values())
            num_jobs = len(futures )
            hosts = list(host2future.keys())
            cols = st.columns(4)
            failed_hosts = []

            try:
                for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):
                    host = hosts[result['idx']]

                    if host == None:
                        continue
                    host2future.pop(host)
                    result = result['result']
                    is_error = c.is_error(result)
                    emoji = c.emoji('cross') if is_error else c.emoji('check_mark')
                    msg = f"""```bash\n{result['error']}```""" if is_error else f"""```bash\n{result}```"""

                    with st.expander(f'{host} -> {emoji}', expanded=False):
                        msg = fn_code(x=msg)
                        if is_error:
                            st.write('ERROR')
                            failed_hosts += [host]
                        if filter_bool and msg != True:
                            continue
                        st.markdown(msg)
                    

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



RemoteDashboard.run(__name__)
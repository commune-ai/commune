import commune as c
import streamlit as st
from typing import *
import json

class Peer(c.Module):
    @classmethod
    def add_host(cls, 
                 cmd:str = None , # in the format of 
                 host:str = '0.0.0.0',
                 port:int = 22,
                 user:str = 'root',
                 pwd:str = None,
                 name : str = None
                 ):
        
        hosts = cls.hosts()
        host = {
            'host': host,
            'port': port,
            'user': user,
            'pwd': pwd
        }
        if name == None:
            cnt = 0
            name = f'{user}{cnt}'

            while name in hosts:
                name = f'{user}{cnt}'
                cnt += 1
        
        hosts[name] = host
        cls.save_hosts(hosts)

        return {'status': 'success', '': f'Host added', }
    
    @classmethod
    def save_hosts(cls, hosts=None, filetype=filetype, path = None):
        if path == None:
            path = cls.host_data_path
        if hosts == None:
            hosts = cls.hosts()
        if filetype == 'json':
            cls.put_json(path, hosts)
        elif filetype == 'yaml':
            cls.put_yaml(path, hosts)

        return {
                'status': 'success', 
                'msg': f'Hosts saved', 
                'hosts': hosts, 
                'path': cls.host_data_path, 
                'filetype': filetype
                }
    @classmethod
    def load_hosts(cls, path = None, filetype=filetype):
        if path == None:
            path = cls.host_data_path
        if filetype == 'json':
            return cls.get_json(path, {})
        elif filetype == 'yaml':
            return cls.get_yaml(path, {})
    
    @classmethod
    def switch_hosts(cls, path):
        hosts = c.get_json(path)
        cls.save_hosts(hosts)
        return {'status': 'success', 'msg': f'Host data path switched to {path}'}
    
    @classmethod
    def rm_host(cls, name):
        hosts = cls.hosts()
        if name in hosts:
            del hosts[name]
            cls.save_hosts( hosts)
            c.print(cls.hosts())
            return {'status': 'success', 'msg': f'Host {name} removed'}
        else:
            return {'status': 'error', 'msg': f'Host {name} not found'}

    @classmethod
    def hosts(cls, search=None, filetype=filetype, enable_search_terms: bool = True):
        hosts = cls.load_hosts(filetype=filetype)
        if len(hosts) == 0:
            assert False, f'No hosts found, please add your hosts to {cls.host_data_path}'
        if search != None:
            hosts = {k:v for k,v in hosts.items() if search in k}

        if enable_search_terms:
            return cls.filter_hosts(hosts=hosts)
        return hosts

    host_map = hosts

    @classmethod
    def host2ip(cls, search=None):
        hosts = cls.hosts(search=search)
        return {k:v['host'] for k,v in hosts.items()}

    @classmethod
    def ip2host(cls, search=None):
        host2ip = cls.host2ip(search=search)
        return {v:k for k,v in host2ip.items()}
    @classmethod
    def names(cls, search=None):
        return list(cls.hosts(search=search).keys())

    def host2name(self, host):
        hosts = self.hosts()
        for name, h in hosts.items():
            if h == host:
                return name
        raise Exception(f'Host {host} not found')
    
    @classmethod
    def n(cls, search=None):
        return len(cls.hosts(search=search))

    def num_servers(self):
        return len(self.servers())
    

    @classmethod
    def n_servers(self):
        return len(self.servers())
    
    @classmethod
    def host(self, name):
        hosts = self.hosts()

        if name not in hosts:
            raise Exception(f'Host {name} not found')
        
        return hosts[name]
    @classmethod
    def has(cls, name):
        return name in cls.hosts()
    @classmethod
    def host_exists(self, name):
        return name in self.hosts()
    @classmethod
    def install(self):
        c.cmd('pip3 install paramiko')
    def test(self):
        # Test Remote
        c.print(self.ssh_cmd('ls'))
    @classmethod
    def cmd(cls, *commands, search=None, hosts:Union[list, dict, str] = None, cwd=None, host:str=None,  timeout=5 , verbose:bool = True, num_trials=1, **kwargs):
        output = {}
        if hosts == None:
            hosts = cls.hosts()
            if host != None:
                hosts = {host:hosts[host]}
        if search != None:
            hosts = {k:v for k,v in hosts.items() if search in k}
        if isinstance(hosts, list):
            hosts = {h:hosts[h] for h in hosts}
        elif isinstance(hosts, str):
            hosts = {hosts:cls.hosts(hosts)}

        assert isinstance(hosts, dict), f'Hosts must be a dict, got {type(hosts)}'

        results = {}
        for host in hosts:
            result_future = c.submit(cls.ssh_cmd, args=commands, kwargs=dict(host=host, cwd=cwd, verbose=verbose,**kwargs), return_future=True)
            results[host] = result_future

        result_values = c.wait(list(results.values()), timeout=timeout)
        results =  dict(zip(results.keys(), result_values))
        results =  {k:v for k,v in results.items()}

        if all([v == None for v in results.values()]):
            raise Exception(f'all results are None')
        
        for k,v in results.items():
            if isinstance(v, str):
                results[k] = v.strip('\n')

        return results 

    @classmethod
    def add_admin(cls, timeout=10):
        root_key_address = c.root_key().ss58_address
        return cls.cmd(f'c add_admin {root_key_address}', timeout=timeout)
    
    @classmethod
    def is_admin(cls, timeout=10):
        root_key_address = c.root_key().ss58_address
        results =  cls.cmd(f'c is_admin {root_key_address}', timeout=timeout)
        for host, r in results.items():
            results[host] = bool(r)
        return results
    
    def add_server(self, address):
        return c.add_server(address, network='remote')
    
    def host2rootkey(self, **kwargs):
        host2rootkey =  self.cmd(f'c root_key_address', **kwargs)
        return {k: v if isinstance(v, str) else None for k,v in host2rootkey.items()}

    @classmethod
    def servers(self,search: str ='module', network='remote'):
        return c.servers(search, network=network)
    
    @classmethod
    def serve(cls, *args, update=False, min_memory:int=10, timeout=10, **kwargs):
        modules = cls.available_peers(update=update, min_memory=min_memory)
        c.print(f'Available modules: {modules}')
        module = c.choice(modules)
        c.print(f'Serving  on {module}')
        namespace = c.namespace(network='remote')
        address = namespace[module]
        module = c.connect(address)
        return module.serve(*args, **kwargs, timeout=timeout)
    
    @classmethod
    def fleet(cls, *args, update=False, min_memory:int=20, n=10, tag=None, **kwargs):
        responses = []
        for i in range(n):
            try:
                response = cls.serve(*args, 
                        update=update, 
                        min_memory=min_memory, 
                        tag= '' if tag == None else tag + str(i), 
                        **kwargs
                        )
            except Exception as e:
                response = c.detailed_error(e)
            c.print(response)
            responses += [response]
        return responses

    @classmethod
    def logs(cls, module, n=3 , **kwargs):
        namespace = cls.namespace(search=module)
        c.print(namespace)
        for name, address in list(namespace.items())[:n]:
            if address == None:
                raise Exception(f'Address for {name} not found')
            logs = c.call(address, 'logs', name, mode='local')
            c.print(f'[bold yellow]{name}[/bold yellow]')
            c.print('\n'.join(logs.split('\n')[-10:]))
        
    
    @classmethod
    def namespace(cls, search=None, network='remote', update=False):
        namespace = {}
        if not update:
            namespace = c.get_namespace(network=network)
            return namespace
        
        peer2namespace = cls.peer2namespace()
        for peer, peer_namespace in peer2namespace.items():

            for name, address in peer_namespace.items():
                if search != None and search not in name:
                    continue
                if name in namespace:
                    continue
                namespace[name + '_'+ {peer}] = address
        c.put_namespace(namespace=namespace, network=network)
        return namespace

    @classmethod
    def get_address(self, name):
        return c.get_address(name)

    
    @classmethod
    def addresses(self, search=None, network='remote'):
        return c.addresses(search=search, network=network)
    
    @classmethod
    def peer2address(self, network='remote'):
        return c.namespace(search='module', network=network)
    
    
    
    def keys(self):
        return [info.get('key', None)for info in self.infos()]
    
    @classmethod
    def infos(self, search='module',  network='remote', update=False):
        return c.infos(search=search, network=network, update=update)


    @classmethod
    def push(cls,**kwargs):
        return [c.push(), cls.pull()]

        
    @classmethod
    def pull(cls, stash=True, hosts=None):
        return c.rcmd(f'c pull stash={stash}', hosts=hosts)
    
    @classmethod
    def hardware(cls, timeout=20, update= True, cache_path:str = 'hardware.json', trials=2):


        if not update:
            peer2hardware = c.get_json(cache_path, {})
            if len(peer2hardware) > 0:
                return peer2hardware
        peer2hardware = {p:None for p in peers}

        peers = cls.peers()
        for i in range(trials):
            call_modules = [p for p in peers if peer2hardware[p] == None]
            response =  cls.call('hardware', timeout=timeout, modules=call_modules)
            for peer, hardware in response.items():
                if isinstance(hardware, dict) and 'memory' in hardware:
                    c.print(f'{peer} {hardware}')
                    peer2hardware[peer] = hardware


        c.put_json(cache_path, peer2hardware)
        return peer2hardware
    

    # peers

    def peers(self, network='remote'):
        return self.servers(search='module', network=network)
    
    def peer2hash(self, search='module', network='remote'):
        return self.call('chash', search=search, network=network)

    def peer2memory(self, min_memory:int=0, **kwargs):
        peer2hardware = self.hardware(**kwargs)
        peer2memory = {}
        for server, hardware in peer2hardware.items():
            memory_info = hardware['memory']
            if isinstance(memory_info, dict) and 'available' in memory_info:
                if memory_info['available'] >= min_memory:
                    peer2memory[server] = memory_info['available']
        return peer2memory
    


    def ps(self, update=False, **kwargs):
        peer2ps = {}
        for peer, ps in self.call('ps', **kwargs).items():
            peer2ps[peer] = ps
        return peer2ps
    
    @classmethod
    def available_peers(cls, min_memory:int=10, update: bool=False, address=False):
        peer2hardware = cls.hardware(update=update)
        peer2memory = {}
        
        for peer, hardware in peer2hardware.items():
            memory_info = hardware['memory']
        
            if isinstance(memory_info, dict) and 'available' in memory_info:
                if memory_info['available'] >= min_memory:
                    peer2memory[peer] = memory_info['available']
        
        if address :
            namespace = c.namespace(network='remote')
            peers = [namespace.get(k) for k in peer2memory]
        else :
            peers = list(peer2memory.keys())
        return peers
    

    @classmethod
    def most_available_peer(cls, min_memory:int=8, update: bool=False):
        peer2hardware = cls.hardware(update=update)
        peer2memory = {}
        
        for peer, hardware in peer2hardware.items():
            memory_info = hardware['memory']
            if isinstance(memory_info, dict) and 'available' in memory_info:
                peer2memory[peer] = memory_info['available']
        
        # get the top n peers with the most memory
        peer2memory = {k:v for k,v in sorted(peer2memory.items(), key=lambda x: x[1], reverse=True)}
        most_available_peer = list(peer2memory.keys())[0]
        most_available_peer_memory = peer2memory[most_available_peer]
        assert most_available_peer_memory >= min_memory, f'Not enough memory, {most_available_peer_memory} < {min_memory}'
        c.print(f'Most available peer: {most_available_peer} with {most_available_peer_memory} GB memory')
        return most_available_peer
    
    @classmethod
    def push(self):
        c.push()
        c.rcmd('c pull', verbose=True)
        c.rcmd('c serve', verbose=True)
        c.add_peers()
    
    @classmethod
    def check_peers(cls, timeout=10):
        futures = []
        for m,a in c.namespace(network='remote').items():
            futures += [c.submit(c.call, args=(a,'info'),return_future=True)]
        results = c.wait(futures, timeout=timeout)
        return results
    
    @classmethod
    def setup(cls,**kwargs):
        repo_url = c.repo_url()
        c.print(cls.cmd(f'git clone {repo_url}', **kwargs))
        c.print(cls.cmd(f'apt ', **kwargs))
        c.print(cls.cmd(f'cd commune && pip install -e .', **kwargs))
        c.print(cls.cmd(f'c add_admin {c.root_key().ss58_address} ', **kwargs))
        c.print(cls.cmd(f'c serve', **kwargs))

    def enter(self, host='root10'):
        host2ssh  = self.host2ssh()
        c.print(host2ssh)
        ssh = host2ssh[host]
        c.cmd(ssh)

    
    def loop(self, timeout=40, interval=30, max_age=360, remote=True, batch_size=10):
        if remote:
            return self.remote_fn('loop',kwargs = locals())
        while True:
            self.sync()
            c.sleep(10)

    def save_ssh_config(self, path="~/.ssh/config"):
        ssh_config = []

        for host_name, host in self.hosts().items():
            ssh_config.append(f'Host {host_name}')
            ssh_config.append(f'  HostName {host["host"]}')
            ssh_config.append(f'  Port {host["port"]}')
            ssh_config.append(f'  User {host["user"]}')

        ssh_config = '\n'.join(ssh_config)

        return c.put_text(path, ssh_config)
    
        
    def text2hosts(self, text, model='model.openai'):
        prompt = {
            'instruciton': 'given the text place into the following format',
            'text': text,
            'format': list(self.hosts().values())[0],
            'output': None
        }
        model = c.module(model)
        return model.generate(c.python2str(prompt))


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
                        result
                        pass
                    else:
                        st.write(result)
        
        t2 = c.time()
        st.write(f'Info took {t2-t1} seconds')



    def sidebar(self, **kwargs):
        with st.sidebar:
            self.filter_hosts_dashboard()
            self.manage_hosts_dashboard()


    # SEARCH TERM LAND

    search_terms_path = 'search_terms'
    @classmethod
    def set_search_terms(cls, search_terms):
        path = cls.search_terms_path
        cls.put(path, search_terms)
        return {'status': 'success', 'msg': f'Search terms set', 'search_terms': search_terms}

    @classmethod
    def clear_terms(cls):
        path = cls.search_terms_path
        return  cls.put(path, {'include': '', 'avoid': ''})


    @classmethod
    def avoid(cls, *terms):
        terms = ','.join(terms)
        search_terms = cls.get_search_terms()
        search_terms['avoid'] = terms
        cls.set_search_terms(search_terms)
        return {'status': 'success', 'msg': f'Added {terms} to avoid terms', 'search_terms': search_terms}
    
    @classmethod
    def include(cls, *terms):
        terms = ','.join(terms)
        search_terms = cls.get_search_terms()
        search_terms['include'] = terms
        cls.set_search_terms(search_terms)
        return {'status': 'success', 'msg': f'Added {terms} to include terms', 'search_terms': search_terms}

    @classmethod
    def get_search_terms(cls):
        path = cls.search_terms_path
        return cls.get(path, {'include': '', 'avoid': ''})
    search_terms = get_search_terms

    @classmethod
    def filter_hosts(cls, include=None, avoid=None, hosts=None):

        host_map = hosts or cls.hosts()
        search_terms = cls.search_terms()
        if avoid != None:
            search_terms['avoid'] = avoid
        if include != None:
            search_terms['include'] = include

        for k, v in search_terms.items():
            # 
            if v == None:
                v = ''

            if len(v) > 0:
                if ',' in v:
                    v = v.split(',')
                else:
                    v = [v]
                
                v = [a.strip() for a in v]
            else:
                v = []
            search_terms[k] = v

        def filter_host(host_name):
            for avoid_term in search_terms["avoid"]:
                if avoid_term in host_name:
                    return False
            for include_term in search_terms["include"]:
                if not include_term in host_name:
                    return False
            return True

        return {k:v for k,v in host_map.items() if filter_host(k)}



    def filter_hosts_dashboard(self, host_names: list = None):

        host_map = self.hosts()
        host_names = list(host_map.keys())

        # get the search terms
        search_terms = self.get_search_terms()
        for k, v  in search_terms.items():
            search_terms[k] = st.text_input(k, v)     
        self.set_search_terms(search_terms)
        host_map = self.filter_hosts(**search_terms)
        host_names = list(host_map.keys())
        n = len(host_names)

        self.filter_hosts()
        

        with st.expander(f'Hosts (n={n})', expanded=False):
            host_names = st.multiselect('Host', host_names, host_names)
        host_map = {k:host_map[k] for k in host_names}
        self.host2ssh = self.host2ssh(host_map=host_map)


    @classmethod
    def host2ssh(cls, host_map=None):
        host_map = host_map or cls.hosts()
        host2ssh = {}
        for k, v in host_map.items():
            host2ssh[k] = f'sshpass -p {v["pwd"]} ssh {v["user"]}@{v["host"]} -p {v["port"]}'
        return host2ssh
    

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
                self.add_host(host=host, port=port, user=user, pwd=pwd)

        with st.expander('Remove Host', expanded=False):
            host_names = list(self.hosts().keys())
            rm_host_name = st.selectbox('Host Name', host_names)
            rm_host = st.button('Remove Host')
            if rm_host:
                self.rm_host(rm_host_name)

        
    
    def ssh_dashboard(self):
        import streamlit as st
        host_map = self.hosts()

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
            filter_bool = cols[2].checkbox('Filter', False)

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
                future = c.submit(self.ssh_cmd, args=[cmd], kwargs=dict(host=host, verbose=False, sudo=sudo, search=host_names, cwd=cwd), return_future=True, timeout=timeout)
                host2future[host] = future

            futures = list(host2future.values())
            num_jobs = len(futures )
            hosts = list(host2future.keys())
            host2error = {}
            cols = st.columns(num_columns)
            failed_hosts = []
            col_idx = 0

            try:
                for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):

                    host = hosts[result['idx']]
                    if host == None:
                        continue

                    host2future.pop(host)
                    result = result['result']
                    is_error = c.is_error(result)
                    emoji = c.emoji('cross') if is_error else c.emoji('check_mark')
                    msg = result['error'] if is_error else result.strip()

                    # get the colkumne
                    col_idx = (col_idx) % len(cols)
                    col = cols[col_idx]
                    col_idx += 1


                    # if the column is full, add a new column
                    with col:
                        with st.expander(f'{host} -> {emoji}', expanded=expanded):
                            msg = fn_code(msg)
                            if is_error:
                                st.write('ERROR')
                                st.error(msg)
                                failed_hosts += [host]
                            else:
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

    @classmethod
    def peer2key(cls, search=None, network:str='remote', update=False):
        infos = c.infos(search=search, network=network, update=update)
        return {v['name']:v['key'] for v in infos if 'name' in v and 'address' in v}

    @classmethod
    def peer_addresses(cls, network:str='remote'):
        infos = c.infos(network=network)
        return {info['key'] for info in infos if 'key' in info}
    
 
    def check_peers(self, timeout=10):
        futures = []
        for m,a in c.namespace(network='remote').items():
            futures += [c.submit(c.call, args=(a,'info'),return_future=True)]
        results = c.wait(futures, timeout=timeout)
        return results
    

    def sync(self, timeout=40,  max_age=360):
        futures = []
        namespace = c.namespace('module', network='remote')
        paths = []
        for name, address in namespace.items():
            path = 'peers/' + name
            existing_peer_info = self.get(path, {})
            peer_update_ts = existing_peer_info.get('timestamp', 0)
            future = c.submit(c.call, 
                                args = [address, 'info'],
                                kwargs = dict(schema=True, namespace=True, hardware=True),
                                timeout=timeout, return_future=True
                                )
            paths += [path]
            futures += [future]

        results = c.wait(futures, timeout=timeout, generator=False)
        for i, result in enumerate(results):
            path = paths[i]
            if c.is_error(result):
                c.print(f'Error {result}')
                continue
            else:
                c.print(f'Success {path}')
                self.put(path, result)
            self.put(path, result)
        return {'status': 'success', 'msg': f'Peers synced'}

    def peerpath2name(self, path:str):
        return path.split('/')[-1].replace('.json', '')
    

    def peer2info(self):
        peer_infos = {}
        for path in self.ls('peers'):
            peer_name = self.peerpath2name(path)
            info = self.get(path, {})   
            peer_infos[peer_name] = info
            peer_infos[peer_name] = info
        return peer_infos
    

    def peer2lag(self, max_age=1000):
        peer2timestamp = self.peer2timestamp()
        time = c.time()
        ip2host = self.ip2host()
        return {ip2host.get(k,k):time - v for k,v in peer2timestamp.items() if time - v < max_age}

    def peer2timestamp(self):
        peer2info = self.peer2info()
        return {k:v.get('timestamp', 0) for k,v in peer2info.items()}

    def peer2hardware(self):
        info_paths = self.ls('peers')
        peer_infos = []
        for path in info_paths:
            info = self.get(path, {})
            # c.print(info)
            peer_infos += [info.get('hardware', {})]
        return peer_infos
    
    @classmethod
    def peer2namespace(cls):
        info_paths = cls.ls('peers')
        peer2namespace = {}
        for path in info_paths:
            info = cls.get(path, {})
            peer2namespace[path] = info.get('namespace', {})
        return peer2namespace



    @classmethod
    def add_peers(cls, add_admins:bool=False, timeout=20, update=True, network='remote'):
        """
        Adds servers to the network by running `c add_peers` on each server.
        
        """
        if update:
            c.rm_namespace(network=network)
        if add_admins:
            cls.add_admin(timeout=timeout)

        namespace = c.namespace(network=network)
        address2name = {v:k for k,v in namespace.items()}
        host2_server_addresses_responses = cls.cmd('c addy', verbose=True, timeout=timeout)
        for i, (host,server_address) in enumerate(host2_server_addresses_responses.items()):
            if isinstance(server_address, str):
                server_address = server_address.split('\n')[-1]

                if server_address in address2name:
                    server_name = address2name[server_address]
                    c.print(f'{server_name} already in namespace')
                    continue
                else:
                    ip = ':'.join(server_address.split(':')[:-1])

                    server_name = 'module' + '_' +  host
                    namespace[server_name] = server_address

        c.put_namespace(network=network, namespace=namespace)

        return {'status': 'success', 'msg': f'Servers added', 'namespace': namespace}
 



    def peer_info(self, peer):
        host2ip = self.host2ip()
        peer = host2ip.get(peer, peer)
        return self.get(f'peers/{peer}', {})
    


    @classmethod
    def call(cls, fn:str='info' , *args, 
             search:str='module', 
             modules=None,  
             network:str='remote',
             avoid_hosts: str = 'root',
               n:int=None, 
               return_future: bool = False, 
               timeout=4, **kwargs):
        futures = {}
        kwargs['network'] =  network
            
        namespace = c.namespace(search=search, network=network)

        if modules != None:
            assert isinstance(modules, list), f'modules must be a list, got {type(modules)}'
            namespace = {k:v for k,v in namespace.items() if k in modules}
        if n == None:
            n = len(namespace)
            
        for name, address in c.shuffle(list(namespace.items()))[:n]:
            c.print(f'Calling {name} {address}')
            futures[name] = c.async_call(address, fn, *args)
        
        if return_future:
            if len(futures) == 1:
                return list(futures.values())[0]
            return futures
        else:

    
            
            num_futures = len(futures)
            results = {}
            import tqdm 


            progress_bar = tqdm.tqdm(total=num_futures)
            error_progress = tqdm.tqdm(total=num_futures)

            results = c.gather(list(futures.values()), timeout=timeout)

            for i, result in enumerate(results):
                if c.is_error(result):
                    # c.print(f'Error {result}')
                    error_progress.update(1)
                    continue

                else:
                    # c.print(f'Success {result}')
                    results[i] = result
                    progress_bar.update(1)
            # if len(results) == 1:
            #     return list(results.values())[0]
        
            return results

        

Peer.run(__name__)

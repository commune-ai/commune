import commune as c
import streamlit as st
from typing import *

class Remote(c.Module):
    filetype = 'yaml'
    host_data_path = f'{c.datapath}/hosts.{filetype}'
    host_url = 'https://raw.githubusercontent.com/communeai/commune/main/hosts.yaml'
    executable_path='commune/bin/c'
    @classmethod
    def ssh_cmd(cls, *cmd_args, host:str= None,  cwd:str=None, verbose=True, sudo=False, key=None, timeout=10,  **kwargs ):
        """s
        Run a command on a remote server using Remote.

        :param host: Hostname or IP address of the remote machine.
        :param port: Remote port (typically 22).
        :param username: Remote username.
        :param password: Remote password.
        :param command: Command to be executed on the remote machine.
        :return: Command output.
        """
        command = ' '.join(cmd_args).strip()
        
        if command.startswith('c '):
            command = command.replace('c ', cls.executable_path + ' ')

        if cwd != None:
            command = f'cd {cwd} && {command}'

        


        import paramiko
        hosts = cls.hosts()
        host_name = host
        if host_name == None:
            host = list(hosts.keys())[0]
        if host_name not in hosts:
            raise Exception(f'Host {host_name} not found')
        host = hosts[host_name]

        # Create an Remote client instance.
        client = paramiko.SSHClient()

        # Automatically add the server's host key (this is insecure and used for demonstration; 
        # in production, you should have the remote server's public key in known_hosts)
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
        # Connect to the remote server
        client.connect(host['host'],
                       port=host['port'], 
                       username=host['user'], 
                       password=host['pwd'])
        
        if sudo and host['user'] != "root":
            command = "sudo -S -p '' %s" % command
        stdin, stdout, stderr = client.exec_command(command)

        try:
            if sudo:
                stdin.write(host['pwd'] + "\n")
                stdin.flush()
            color = c.random_color()
            # Print the output of ls command
            outputs = {'error': '', 'output': ''}

            for line in stdout.readlines():
                if verbose:
                    c.print(f'[bold]{host_name}[/bold]', line.strip('\n'), color=color)
                outputs['output'] += line

            for line in stderr.readlines():
                if verbose:
                    c.print(f'[bold]{host_name}[/bold]', line.strip('\n'))
                outputs['error'] += line
        
            if len(outputs['error']) == 0:
                outputs = outputs['output']
    
            stdin.close()
            stdout.close()
            stderr.close()
            client.close()
        except Exception as e:
            c.print(e)
            pass
        return outputs

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

        return {'status': 'success', 
                'msg': f'Hosts saved', 
                'hosts': hosts, 
                'path': cls.host_data_path, 
                'filetype': filetype}
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
    def hosts(cls, search=None, filetype=filetype):
        hosts = cls.load_hosts(filetype=filetype)
        if len(hosts) == 0:
            assert False, f'No hosts found, please add your hosts to {cls.host_data_path}'
        if search != None:
            hosts = {k:v for k,v in hosts.items() if search in k}
        return hosts


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
    def cmd(cls, *commands, hosts:Union[list, dict, str] = None, cwd=None, host:str=None,  timeout=5 , verbose:bool = True, num_trials=1, **kwargs):

        output = {}
        if hosts == None:
            hosts = cls.hosts()
            if host != None:
                hosts = {host:hosts[host]}
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
    
    @classmethod
    def add_servers(cls, add_admins:bool=False, timeout=20, refresh=False, network='remote'):
        """
        Adds servers to the network by running `c add_servers` on each server.
        
        """
        if refresh:
            c.rm_namespace(network=network)
        if add_admins:
            cls.add_admin(timeout=timeout)

        namespace = c.namespace(network=network)
        address2name = {v:k for k,v in namespace.items()}
        ip2host = cls.ip2host()


    
        server_addresses_responses = list(cls.cmd('c addy', verbose=True, timeout=timeout).values())
        for i, server_address in enumerate(server_addresses_responses):
            if isinstance(server_address, str):
                server_address = server_address.split('\n')[-1]

                if server_address in address2name:
                    server_name = address2name[server_address]
                    c.print(f'{server_name} already in namespace')
                    continue
                else:
                    ip = c.address2ip(server_address)
                    if ip in ip2host:
                        host = ip2host[ip]
                    server_name = 'module' + '_' +  str(host)
                    namespace[server_name] = server_address

        c.put_namespace(network=network, namespace=namespace)
        return {'status': 'success', 'msg': f'Servers added', 'servers': namespace}

    @classmethod
    def servers(self,search: str ='module', network='remote'):
        return c.servers(search, network=network)
    
    @classmethod
    def serve(cls, *args, update=False, min_memory:int=10, timeout=10, **kwargs):
        modules = cls.available_servers(update=update, min_memory=min_memory)
        module = c.choice(modules)
        c.print(f'Serving  on {module}')
        module = c.connect(module, network='remote')
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

        if update:
            namespace = {}
            host2namespace = cls.call('namespace', public=True, timeout=20)

            for host, host_namespace in host2namespace.items():
                if c.is_error(host_namespace):
                    continue
                for name, address in host_namespace.items():
                    tag = ''
                    while name + str(tag) in namespace:
                        if tag == '':
                            tag = 1
                        else:
                            tag += 1
                    namespace[name + str(tag)] = address
            c.put_namespace(namespace=namespace, network=network)
        else:
            namespace = c.get_namespace(search, network=network)
        if search != None: 
            namespace = {k:v for k,v in namespace.items() if search in k}

        address2name = {v:k for k,v in namespace.items()}
        ip2host = cls.ip2host()
        local_ip = c.ip()
        for address, name in address2name.items():
            if 'module' in name:
                if address in address2name:
                    continue
                else:
                    ip = c.address2ip(address)                        

                    if ip in ip2host:
                        host = ip2host[ip]
                    server_name = 'module' + '_' +  str(host)
                    namespace[server_name] = address
        return namespace

    @classmethod
    def get_address(self, name):
        return c.get_address(name)

    
    @classmethod
    def addresses(self, search=None, network='remote'):
        return c.addresses(search=search, network=network)
    
    
    
    def keys(self):
        return [info.get('ss58_address', None)for info in self.server_infos()]
    
    @classmethod
    def server_infos(self, search=None,  network='remote', update=False):
        return c.server_infos(search=search, network=network, update=update)
    @classmethod
    def server2key(cls, search=None, network:str='remote', update=False):
        infos = c.server_infos(search=search, network=network, update=update)
        return {v['name']:v['ss58_address'] for v in infos if 'name' in v and 'address' in v}

    @classmethod
    def key_addresses(cls, network:str='remote'):
        infos = c.server_infos(network=network)
        return {info['ss58_address'] for info in infos if 'ss58_address' in info}
    @classmethod
    def push(cls,**kwargs):
        return [c.push(), cls.pull()]


    @classmethod
    def call(cls, fn:str='info' , *args, search:str='module', module=None,  network:str='remote', n:int=None, return_future: bool = False, timeout=20, **kwargs):
        futures = {}
        kwargs['network'] =  network
            
        namespace = c.namespace(search=search, network=network)
        if n == None:
            n = len(namespace)
            
        for name, address in c.shuffle(list(namespace.items()))[:n]:
            c.print(f'Calling {name} {address}')
            futures[name] = c.submit(c.call, args=(address, fn, *args), kwargs=kwargs, return_future=True, timeout=timeout)
        
        if return_future:
            if len(futures) == 1:
                return list(futures.values())[0]
            return futures
        else:

            results = c.wait(list(futures.values()), timeout=timeout)
            results = dict(zip(futures.keys(), results))
            # if len(results) == 1:
            #     return list(results.values())[0]
        
            return results

        
        
    @classmethod
    def pull(cls, stash=True, hosts=None):
        return c.rcmd(f'c pull stash={stash}', hosts=hosts)
    
    @classmethod
    def hardware_info(cls, timeout=10, update= False, cache_path:str = 'hardware_info.json'):

        if not update:
            server2hardware =  c.get_json(cache_path, {})
            if len(server2hardware) > 0:
                return server2hardware
    

        server2hardware_info =  cls.call('hardware_info', timeout=timeout)
        server2hardware = {}
        for server, info in server2hardware_info.items():
            if isinstance(info, dict) and 'memory' in info:
                server2hardware[server] = info
        c.put_json(cache_path, server2hardware)
        return server2hardware
    

    def peers(self, network='remote'):
        return self.servers(search='module', network=network)
    

    def server2memory(self, min_memory:int=10, **kwargs):
        server2hardware_info = self.hardware_info(**kwargs)
        server2memory = {}
        
        for server, hardware_info in server2hardware_info.items():
            memory_info = hardware_info['memory']
        
            if isinstance(memory_info, dict) and 'available' in memory_info:
                if memory_info['available'] >= min_memory:
                    server2memory[server] = memory_info['available']

        return server2memory
    
    @classmethod
    def available_servers(cls, min_memory:int=10, update: bool=False, address=False):
        server2hardware_info = cls.hardware_info(update=update)
        server2memory = {}
        
        for server, hardware_info in server2hardware_info.items():
            memory_info = hardware_info['memory']
        
            if isinstance(memory_info, dict) and 'available' in memory_info:
                if memory_info['available'] >= min_memory:
                    server2memory[server] = memory_info['available']
        
        if address :
            namespace = c.namespace(network='remote')
            servers = [namespace.get(k) for k in server2memory]
        else :
            servers = list(server2memory.keys())
        return servers
    

    @classmethod
    def most_available_server(cls, min_memory:int=8, update: bool=False):
        server2hardware_info = cls.hardware_info(update=update)
        server2memory = {}
        
        for server, hardware_info in server2hardware_info.items():
            memory_info = hardware_info['memory']
            if isinstance(memory_info, dict) and 'available' in memory_info:
                server2memory[server] = memory_info['available']
        
        # get the top n servers with the most memory
        server2memory = {k:v for k,v in sorted(server2memory.items(), key=lambda x: x[1], reverse=True)}
        most_available_server = list(server2memory.keys())[0]
        most_available_server_memory = server2memory[most_available_server]
        assert most_available_server_memory >= min_memory, f'Not enough memory, {most_available_server_memory} < {min_memory}'
        c.print(f'Most available server: {most_available_server} with {most_available_server_memory} GB memory')
        return most_available_server
    
    @classmethod
    def push(self):
        c.push()
        c.rcmd('c pull', verbose=True)
        c.rcmd('c serve', verbose=True)
        c.add_servers()
    
    @classmethod
    def check_servers(cls, timeout=10):
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

    def sidebar(self):
        import streamlit as st

        with st.sidebar:
            with st.expander('Add Host', expanded=False):
                st.markdown('## Hosts')
                cols = st.columns(2)
                host = cols[0].text_input('Host',  '0.0.0.0')
                port = cols[1].number_input('Port', 22, 10000, 22)
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



    @classmethod
    def dashboard(cls, deploy:bool=True):

        if deploy:
            cls.st(kwargs=dict(deploy=False))
        self = cls()

        import streamlit as st

        st.set_page_config(layout="wide")
        st.title('Remote Dashboard')
        self.sidebar()

        self.st = c.module('streamlit')()
        self.st.load_style()

        tabs = st.tabs(['Servers', 'Modules'])


        with tabs[0]:
            st.markdown('## Servers')
            self.ssh_dashboard()
        with tabs[1]:
            st.markdown('## Servers')
            self.servers_dashboard()


    def servers_dashboard(self):
        import streamlit as st

        cols = st.columns(2)
        search = cols[0].text_input('Search', 'module')
        namespace = c.namespace(search=search, network='remote')
        n = cols[1].number_input('Number of servers', 1, len(namespace), 10)
        module_names = list(namespace.keys())
        module_names = st.multiselect('Modules', module_names, module_names)
        namespace = {k:v for k,v in namespace.items() if k in module_names}
        module_addresses = list(namespace.values())
        
        module_names = list(namespace.keys())
        if len(module_names) == 0:
            st.error('No modules found')
            return
        
        
        module_name = st.selectbox('Module', module_names, index=0)
        module_address = namespace[module_name]
        module = c.connect(module_address)

        cache = st.checkbox('Cache')
        cache_path = f'module_info_cache/{module_address}'
        t1 = c.time()
        if cache:
            module_info = self.get_json(cache_path, {})
        else:
            module_info = {}

        if len(module_info) == 0:
            module_info = module.info()
            self.put_json(cache_path, module_info)

        fns = list(module_info['schema'].keys())
        fn_name = st.selectbox('Function', fns, index=0)

        module_fn = getattr(module, fn_name)
        

        kwargs = self.function2streamlit(fn=fn_name, fn_schema=module_info['schema'][fn_name])


        run = st.button(f'Run {fn_name}')
        if run:
            future2module = {}
            for module_address in module_addresses:
                future = c.submit(c.call, args=[module_address], kwargs=kwargs, return_future=True)
                future2module[future] = module_address
            futures = list(future2module.values())
            import concurrent
            kwargs['return_future'] = True
            result = module_fn(**kwargs)
            st.write(result)
        
        t2 = c.time()
        st.write(f'Info took {t2-t1} seconds')





    def ssh_dashboard(self):
        import streamlit as st
        host_map = self.hosts()
        cols = st.columns(2)
        host_names = list(host_map.keys())

            
        search = st.text_input('Search')


        if len(search) > 0:
            host_names = [h for h in host_names if search in h]
        hosts = self.hosts()
        hosts = {k:v for k,v in hosts.items() if k in host_names}
        host_names = list(hosts.keys())

        with st.expander('Hosts', expanded=False):
            for host_name, host in hosts.items():
                cols = st.columns([1,4,2])
                cols[0].write('#### '+host_name)
                cols[1].code(f'sshpass -p {host["pwd"]} ssh {host["user"]}@{host["host"]} -p {host["port"]}')
                remove_host  = cols[2].button(f'Remove {host_name}')
                if remove_host:
                    st.write(self.rm_host(host_name))
                
        host_names = st.multiselect('Host', host_names, host_names)
        cols = st.columns([4,2,1,1])


        cmd = cols[0].text_input('Command', 'ls')

        [cols[1].write('') for i in range(2)]
        run_button = cols[1].button('Run')
        timeout = cols[2].number_input('Timeout', 1, 100, 10)
        # add splace to cols[2] vertically
        [cols[3].write('') for i in range(2)]
        sudo = cols[3].checkbox('Sudo')

        host2future = {}
        if run_button:
            for host in host_names:
                future = c.submit(self.ssh_cmd, args=[cmd], kwargs=dict(host=host, verbose=False, sudo=sudo, search=host_names), return_future=True, timeout=timeout)
                host2future[host] = future

        futures = list(host2future.values())
        hosts = list(host2future.keys())
        host2error = {}
        try:
            for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):
                host = hosts[result['idx']]
                if host == None:
                    continue
                host2future.pop(host)

                result = result['result']
                if c.is_error(result):
                    host2error[host] = result
                else:
                    st.markdown(host + ' ' + c.emoji('check_mark'))
                    st.markdown(f"""```bash\n{result}```""")

        except Exception as e:
            pending_hosts = list(host2future.keys())
            st.error(c.detailed_error(e))
            st.error(f"Hosts {pending_hosts} timed out")

        for host, result in host2error.items():
            st.markdown(host + ' ' + c.emoji('cross'))
            st.markdown(f"""```bash\n{result}```""")



    def reg_servers(self, search='vali::cc', stake:str=200, timeout=40, batch_size=10):
        namespace = self.namespace(search=search)
        subspace = c.module('subspace')()
        launcher_keys = subspace.launcher_keys()
        c.print(f'Registering with launcher keys {len(launcher_keys)} for {len(namespace)} servers')
        subspace_namespace = subspace.namespace(update=True)

        for i, (name, address) in enumerate(namespace.items()):
            if name in subspace_namespace:
                c.print(f'{name} already registered')
                continue
            if len(launcher_keys) == 0:
                c.print(f'No more launcher keys')
                break
            key = launcher_keys.pop(i % len(launcher_keys))
            c.print(f'Registering {name} {address} {key}')
            try:
                result = subspace.register(name=name, address=address, stake=stake, key=key)
            except Exception as e:
                result = c.detailed_error(e)
            c.print(result)

            
        


      

        
        



       

    dash = dashboard








    # @classmethod
    # def refresh_servers(cls):
    #     cls.cmd('')
    
Remote.run(__name__)

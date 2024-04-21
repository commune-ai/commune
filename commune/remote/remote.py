import commune as c
import streamlit as st
from typing import *
import json
import paramiko

class Remote(c.Module):
    filetype = 'yaml'
    host_data_path = f'{c.datapath}/hosts.{filetype}'
    host_url = 'https://raw.githubusercontent.com/communeai/commune/main/hosts.yaml'
    executable_path='commune/bin/c'
    @classmethod
    def ssh_cmd(cls, *cmd_args, 
                
                port = None, 
                user = None,
                password = None,
                host:str= None,  
                cwd:str=None, 
                verbose=False, 
                sudo=False, 
                key=None, 
                timeout=10,  
                key_policy = 'auto_add_policy',
                **kwargs ):
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
        

        if host == None:
            host = {
                'host': host,
                'port': port,
                'user': user,
                'pwd': password,
            }
        else:
            host = cls.hosts().get(host, None)
            
        
        host['name'] = f'{host["user"]}@{host["host"]}:{host["port"]}'


        c.print(f'Running command: {command} on {host["name"]}')

        # Create an Remote client instance.
        client = paramiko.SSHClient()
        # Automatically add the server's host key (this is insecure and used for demonstration; 
        # in production, you should have the remote server's public key in known_hosts)
        if key_policy == 'auto_add_policy':
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            client.load_system_host_keys()
                
        # Connect to the remote server
        client.connect(host['host'],
                       port=host['port'], 
                       username=host['user'], 
                       password=host['pwd'])
        


        # THE COMMAND

        if cwd != None:
            command = f'cd {cwd} && {command}'

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
                    c.print(f'[bold]{host["name"]}[/bold]', line.strip('\n'), color=color)
                outputs['output'] += line

            for line in stderr.readlines():
                if verbose:
                    c.print(f'[bold]{host["name"]}[/bold]', line.strip('\n'))
                outputs['error'] += line
        
            if len(outputs['error']) == 0:
                outputs = outputs['output']
    
            # stdin.close()
            # stdout.close()
            # stderr.close()
            # client.close()
        except Exception as e:
            c.print(e)
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
    def hosts(cls, search=None, filetype=filetype, enable_search_terms: bool = False):
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
    def cmd(cls, *commands, 
            search=None, 
            hosts:Union[list, dict, str] = None, 
            cwd=None,
              host:str=None,  
              timeout=5 , 
              verbose:bool = True,**kwargs):
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
        errors = {}
        host2future = {}
        for host in hosts:
            host2future[host] = c.submit(cls.ssh_cmd, 
                                            args=commands, 
                                            kwargs=dict(host=host, cwd=cwd, verbose=verbose,**kwargs),
                                            return_future=True
                                            )
        future2host = {v:k for k,v in host2future.items()}


        try:
            for future in c.as_completed(list(host2future.values()), timeout=timeout):
                result = future.result()
                host = future2host[future]
                if not c.is_error(result):
                    results[host] = result
                else:
                    errors[host]= result
        except Exception as e:
            c.print(e)


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
        return [info.get('ss58_address', None)for info in self.infos()]
    
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

    def pwds(self, search=None):
        return {k:v['pwd'] for k,v in self.hosts(search=search).items()}

    @classmethod
    def host2ssh(cls, search = None, host_map=None):
        host_map = host_map or cls.host_map(search=search)
        c.print()
        host2ssh = {}
        for k, v in host_map.items():
            host2ssh[k] = f'sshpass -p {v["pwd"]} ssh {v["user"]}@{v["host"]} -p {v["port"]}'
        return host2ssh
    
    @classmethod
    def peer2key(cls, search=None, network:str='remote', update=False):
        infos = c.infos(search=search, network=network, update=update)
        return {v['name']:v['ss58_address'] for v in infos if 'name' in v and 'address' in v}

    @classmethod
    def peer_addresses(cls, network:str='remote'):
        infos = c.infos(network=network)
        return {info['ss58_address'] for info in infos if 'ss58_address' in info}
    
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
            c.print(path)
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
    def add_peers(cls, add_admins:bool=False, timeout=20, update=False, network='remote'):
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
    @classmethod
    def app(cls):
        return c.module('remote.app').app()

    def save_ssh_config(self, path="~/.ssh/config"):
        ssh_config = self.ssh_config()
        return c.put_text(path, ssh_config) 

    def ssh_config(self, search=None):
        """
        Host {name}
          HostName 0.0.0.0.0
          User fam
          Port 8888
        """
        host_map = self.host_map(search=search)
        toml_text = ''
        for k,v in host_map.items():
            toml_text += f'Host {k}\n'
            toml_text += f'  HostName {v["host"]}\n'
            toml_text += f'  User {v["user"]}\n'
            toml_text += f'  Port {v["port"]}\n'
        
        return toml_text



    @classmethod
    def add_host_from_ssh_string(cls, ssh_string: str, name: str = None):
        """
        Adds a host using an SSH connection string format that includes the password using the -pwd flag.

        :param ssh_string: SSH connection string, e.g., "user@host:port -p ssh_port -pwd password"
        :param name: Optional name for the host; if not provided, a name will be generated
        """
        # Regular expression to parse the SSH connection string including the password specified by -pwd flag
        pattern = r'(?P<user>[^@]+)@(?P<host>[^:]+):(?P<port>\d+).*?-p\s*(?P<ssh_port>\d+).*?-pwd\s*(?P<pwd>[^\s]+)'
        match = re.match(pattern, ssh_string)
        if not match:
            raise ValueError("SSH string format is invalid. Expected format: 'user@host:port -p ssh_port -pwd password'")

        user = match.group('user')
        pwd = match.group('pwd')
        host = match.group('host')
        # The port in the SSH string is not used for SSH connections in this context, so it's ignored
        ssh_port = int(match.group('ssh_port'))

        # Use the existing add_host method to add the host
        return cls.add_host(host=host, port=ssh_port, user=user, pwd=pwd, name=name)
    
    def pwd(self, host):
        hosts = self.hosts(search=host)
        if host not in hosts:
            return {k:v['pwd'] for k,v in hosts.items()}
        return self.hosts()[host]['pwd']
    



Remote.run(__name__)

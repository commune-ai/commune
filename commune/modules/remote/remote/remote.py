import commune as c
import streamlit as st
from typing import *
import os
import re
import json
import paramiko
    
class Remote:
    def __init__(self, path = c.abspath('~/.commune/remote/hosts.yaml')):
        self.path = path

    def ssh_cmd(self, *cmd_args, 
                cmd : str = None,
                port = None, 
                user = None,
                password = None,
                host:str= None,  
                cwd:str=None, 
                verbose=False, 
                sudo=False, 
                stream = False,
                key=None, 
                timeout=10,  
                container = None,
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

        if host == None:
            if port == None or user == None or password == None:
                host = list(self.hosts().values())[0]
            else:
                host = {
                    'host': host,
                    'port': port,
                    'user': user,
                    'pwd': password,
                }
        else:
            host = self.hosts().get(host, None)
        

        host['name'] = f'{host["user"]}@{host["host"]}:{host["port"]}'

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

        command = ' '.join(cmd_args).strip() if cmd == None else cmd


        if cwd != None:
            command = f'cd {cwd} && {command}'

        # Execute the command on the remote server
        if sudo and host['user'] != "root":
            command = "sudo -S -p '' %s" % command
            
        if container != None:
            command = f'docker exec {container} {command}'

        c.print(f'Running --> (command={command} host={host["name"]} sudo={sudo} cwd={cwd})')

        stdin, stdout, stderr = client.exec_command(command)
        try:
            if sudo:
                stdin.write(host['pwd'] + "\n") # Send the password for sudo commands
                stdin.flush() # Send the password
            color = c.random_color()
            # Print the output of ls command
            def print_output():
                for line in stdout.readlines():
                    if verbose:
                        c.print(f'[bold]{host["name"]}[/bold]', line.strip('\n'), color=color)
                    yield line 
                # if there is an stderr, print it

                for line in stderr.readlines():
                    if verbose:
                        c.print(f'[bold]{host["name"]}[/bold]', line.strip('\n'))
                    yield line

            if stream:
                return print_output()
            else:
                output = ''
                for line in print_output():
                    output += line 


            del stdin, stdout, stderr, client

        except Exception as e:
            c.print(e)

        return output

    
    def add_host(self, 
                 host:str = '0.0.0.0',
                 ip = None,
                 port:int = 22,
                 user:str = 'root',
                 pwd:str = None,
                 password:str = None,
                 name : str = None,
                 metadata = None
                 
                 ):
        
        host = ip or host

        if 'ssh ' in host:
            return self.add_host_from_ssh(host)
        
        hosts = self.hosts()
        host = {
            'host': host, # IP address of the remote machine
            'port': port, # Remote port (typically 22)
            'user': user, # Remote username
            'pwd': pwd or password, # Remote password
            'metadata': metadata or {}
        }

        if name == None:
            # 
            cnt = 0
            name = f'{user}{cnt}'
            while name in hosts:
                cnt += 1
                name = f'{user}{cnt}'
        hosts[name] = host
        self.save_hosts(hosts)
        return {'status': 'success', 'msg': f'Host {name} added', 'host': host, 'name': name}
    
    
    def save_hosts(self, hosts=None, path = None):
        if path == None:
            path = self.path
        
        c.print(f'Saving hosts to {path}')
        if hosts == None:
            hosts = self.hosts()
        c.put_yaml(path, hosts)

        return {
                'status': 'success', 
                'msg': f'Hosts saved', 
                'hosts': hosts, 
                'path': self.path, 
                }
    def load_hosts(self, path = None):
        if path == None:
            path = self.path
        if not os.path.exists(path):
            c.print(f'Hosts file {path} does not exist, creating a new one')
            self.save_hosts(hosts={})
        hosts =  c.get_yaml(path)
        if not isinstance(hosts, dict):
            return {}
        return hosts
    
    def switch_hosts(self, path):
        hosts = c.get_json(path)
        self.save_hosts(hosts)
        return {'status': 'success', 'msg': f'Host data path switched to {path}'}
    
    
    def rm_host(self, name):
        hosts = self.hosts()
        if name in hosts:
            del hosts[name]
            self.save_hosts( hosts)
            c.print(self.hosts())
            return {'status': 'success', 'msg': f'Host {name} removed'}
        else:
            return {'status': 'error', 'msg': f'Host {name} not found'}

    def rm_hosts(self, *hosts):
        """
        Remove multiple hosts by name.
        """
        og_hosts = self.hosts()
        for host in hosts:
            og_hosts.pop(host, None)
        self.save_hosts(hosts)
        return {'status': 'success', 'msg': f'Hosts {hosts} removed'}

    def hosts(self, search=None, enable_search_terms: bool = False, path=None):
        hosts = self.load_hosts(path=path)
        if len(hosts) == 0:
            return {}
        
        
        if search != None:
            hosts = {k:v for k,v in hosts.items() if search in k}
        host2ssh = self.host2ssh(host_map=hosts)
        for host, ssh in host2ssh.items():
             hosts[host]['ssh'] = ssh
        if enable_search_terms:
            return self.filter_hosts(hosts=hosts)
        return hosts

    host_map = hosts

    def host2ip(self, search=None):
        hosts = self.hosts(search=search)
        return {k:v['host'] for k,v in hosts.items()}

    def ip2host(self, search=None):
        host2ip = self.host2ip(search=search)
        return {v:k for k,v in host2ip.items()}
    
    def names(self, search=None):
        return list(self.hosts(search=search).keys())

    def host2name(self, host):
        hosts = self.hosts()
        for name, h in hosts.items():
            if h == host:
                return name
        raise Exception(f'Host {host} not found')
    
    def n(self, search=None):
        return len(self.hosts(search=search))

    
    def host(self, name):
        hosts = self.hosts()

        if name not in hosts:
            raise Exception(f'Host {name} not found')
        
        return hosts[name]
    
    def install(self):
        c.cmd('pip3 install paramiko')
    def test(self):
        # Test Remote
        c.print(self.ssh_cmd('ls'))

    def cmd(self, *commands, 
            search=None, 
            hosts:Union[list, dict, str] = None, 
            cwd=None,
              host:str=None,  
              timeout=5 , 
              verbose:bool = True,**kwargs):
        if hosts == None:
            hosts = self.hosts()
            if host != None:
                hosts = {host:hosts[host]}
        if search != None:
            hosts = {k:v for k,v in hosts.items() if search in k}
        if isinstance(hosts, list):
            hosts = {h:hosts[h] for h in hosts}
        elif isinstance(hosts, str):
            hosts = {hosts:self.hosts(hosts)}

        assert isinstance(hosts, dict), f'Hosts must be a dict, got {type(hosts)}'

        results = {}
        errors = {}
        host2future = {}
        for host in hosts:
            host2future[host] = c.submit(self.ssh_cmd, 
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

    def add_admin(self, timeout=10):
        root_key_address = c.root_key().ss58_address
        return self.cmd(f'c add_admin {root_key_address}', timeout=timeout)
    
    def is_admin(self, timeout=10):
        root_key_address = c.root_key().ss58_address
        results =  self.cmd(f'c is_admin {root_key_address}', timeout=timeout)
        for host, r in results.items():
            results[host] = bool(r)
        return results
    
    def logs(self, module, n=3 , **kwargs):
        namespace = self.namespace(search=module)
        c.print(namespace)
        for name, address in list(namespace.items())[:n]:
            if address == None:
                raise Exception(f'Address for {name} not found')
            logs = c.call(address, 'logs', name, stream=False)
            c.print(f'[bold yellow]{name}[/bold yellow]')
            c.print('\n'.join(logs.split('\n')[-10:]))

    def push(self,**kwargs):
        return [c.push(), self.pull()]

        
    def pull(self, stash=True, hosts=None):
        return c.rcmd(f'c pull stash={stash}', hosts=hosts)


    def push(self):
        c.push()
        c.rcmd('c pull', verbose=True)
        c.rcmd('c serve', verbose=True)
        c.add_peers()

    def setup(self,**kwargs):
        repo_url = c.repo_name_url()
        c.print(self.cmd(f'git clone {repo_url}', **kwargs))
        c.print(self.cmd(f'apt ', **kwargs))
        c.print(self.cmd(f'cd commune && pip install -e .', **kwargs))
        c.print(self.cmd(f'c add_admin {c.root_key().ss58_address} ', **kwargs))
        c.print(self.cmd(f'c serve', **kwargs))

    def enter(self, host='root10'):
        host2ssh  = self.host2ssh()
        c.print(host2ssh)
        ssh = host2ssh[host]
        c.cmd(ssh)

    
        
        
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

    def set_search_terms(self, search_terms):
        path = self.search_terms_path
        c.put(path, search_terms)
        return {'status': 'success', 'msg': f'Search terms set', 'search_terms': search_terms}

    def clear_terms(self):
        path = self.search_terms_path
        return  c.put(path, {'include': '', 'avoid': ''})

    def avoid(self, *terms):
        terms = ','.join(terms)
        search_terms = self.get_search_terms()
        search_terms['avoid'] = terms
        self.set_search_terms(search_terms)
        return {'status': 'success', 'msg': f'Added {terms} to avoid terms', 'search_terms': search_terms}
    
    def include(self, *terms):
        terms = ','.join(terms)
        search_terms = self.get_search_terms()
        search_terms['include'] = terms
        self.set_search_terms(search_terms)
        return {'status': 'success', 'msg': f'Added {terms} to include terms', 'search_terms': search_terms}

    def get_search_terms(self):
        path = self.search_terms_path
        return c.get(path, {'include': '', 'avoid': ''})
    search_terms = get_search_terms

    def filter_hosts(self, include=None, avoid=None, hosts=None):

        host_map = hosts or self.hosts()
        search_terms = self.search_terms()
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

    def host2ssh(self, search = None, host_map=None):
        host_map = host_map or self.host_map(search=search)
        c.print()
        host2ssh = {}
        for k, v in host_map.items():
            host2ssh[k] = f'sshpass -p {v["pwd"]} ssh {v["user"]}@{v["host"]} -p {v["port"]}'
        return host2ssh

    
    def call(self, fn:str='info' , *args, 
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

    def app(self):
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


    def add_host_from_ssh(self, ssh: str, name: str):
        """
        Adds a host using an SSH connection string format that includes the password using the -pwd flag.

        :param ssh: SSH connection string, e.g., "user@host -p {port} -pwd password"
        :param name: Optional name for the host; if not provided, a name will be generated
        """
        # Regular expression to parse the SSH connection string including the password specified by -pwd flag
        pattern = r'(?P<user>[^@]+)@(?P<host>[^\s]+)(?:\s+-p\s+(?P<port>\d+))?.*?\s+-pwd\s+(?P<pwd>[^\s]+)'
        match = re.match(pattern, ssh)
        if not match:
            raise ValueError("SSH string format is invalid. Expected format: 'user@host -p {port} -pwd password'")

        user = match.group('user')
        pwd = match.group('pwd')
        host = match.group('host')
        # Default port to 22 if not specified
        port = int(match.group('port')) if match.group('port') else 22

        # Use the existing add_host method to add the host
        return self.add_host(host=host, port=port, user=user, pwd=pwd, name=name)
    
    def pwd(self, host):
        hosts = self.hosts(search=host)
        if host not in hosts:
            return {k:v['pwd'] for k,v in hosts.items()}
        return self.hosts()[host]['pwd']
    def app(self):
        app_path = c.filepath('remote.app')
        return c.cmd(f'streamlit run {app_path}')
    



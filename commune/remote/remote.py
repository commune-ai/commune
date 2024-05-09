import commune as c
import streamlit as st
from typing import *
import os
import json

class Remote(c.Module):
    filetype = 'yaml'
    host_data_path = f'{c.datapath}/hosts.{filetype}'
    host_url = 'https://raw.githubusercontent.com/communeai/commune/main/hosts.yaml'
    executable_path='commune/bin/c'
    @classmethod
    def ssh_cmd(cls, *cmd_args, 
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

        import paramiko
        
        if host == None:
            if port == None or user == None or password == None:
                host = list(cls.hosts().values())[0]
            else:
                host = {
                    'host': host,
                    'port': port,
                    'user': user,
                    'pwd': password,
                }
        else:
            host = cls.hosts().get(host, None)
        

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
                cnt = 0
                for line in stderr.readlines():
                    if cnt == 0:
                        yield '---- ERROR ----'
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

    @classmethod
    def add_host(cls, 
                 host:str = '0.0.0.0',
                 port:int = 22,
                 user:str = 'root',
                 pwd:str = None,
                 password:str = None,
                 name : str = None
                 
                 ):
        
        hosts = cls.hosts()
        host = {
            'host': host, # IP address of the remote machine
            'port': port, # Remote port (typically 22)
            'user': user, # Remote username
            'pwd': pwd or password # Remote password
        }

        if name == None:
            # 
            cnt = 0
            name = f'{user}{cnt}'
            while name in hosts:
                cnt += 1
                name = f'{user}{cnt}'
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

    
    @classmethod
    def host(self, name):
        hosts = self.hosts()

        if name not in hosts:
            raise Exception(f'Host {name} not found')
        
        return hosts[name]
    
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
    def push(cls,**kwargs):
        return [c.push(), cls.pull()]

        
    @classmethod
    def pull(cls, stash=True, hosts=None):
        return c.rcmd(f'c pull stash={stash}', hosts=hosts)
    


    @classmethod
    def push(self):
        c.push()
        c.rcmd('c pull', verbose=True)
        c.rcmd('c serve', verbose=True)
        c.add_peers()


    

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

import commune as c

class Remote(c.Module):
    host_data_path = f'{c.datapath}/hosts.json'
    @classmethod
    def ssh_cmd(cls, *cmd_args, host:str= None,  cwd:str=None, verbose=False,  **kwargs ):
        """s
        Run a command on a remote server using Remote.

        :param host: Hostname or IP address of the remote machine.
        :param port: Remote port (typically 22).
        :param username: Remote username.
        :param password: Remote password.
        :param command: Command to be executed on the remote machine.
        :return: Command output.
        """
        command = ' '.join(cmd_args)

        if cwd != None:
            command = f'cd {cwd} ; {command}'


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
        


        # Execute command
        stdin, stdout, stderr = client.exec_command(command)
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

        return outputs


    @classmethod
    def add_host(cls, 
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
    def save_hosts(cls, hosts):
        cls.put_json(cls.host_data_path, hosts)
    @classmethod
    def load_hosts(cls):
        return cls.get_json(cls.host_data_path, {})
    
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
            cls.put_json(cls.host_data_path, hosts)
            return {'status': 'success', 'msg': f'Host {name} removed'}
        else:
            return {'status': 'error', 'msg': f'Host {name} not found'}

    @classmethod
    def hosts(cls, search=None):
        hosts = cls.get_json(cls.host_data_path, {})
        if len(hosts) == 0:
            assert False, f'No hosts found, please add your hosts to {cls.host_data_path}'
        if search != None:
            hosts = {k:v for k,v in hosts.items() if search in k}
        return hosts

    @classmethod
    def names(cls, search=None):
        return list(cls.hosts(search=search).keys())

    

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
    def cmd(cls, *commands,  search=None, hosts=None, cwd=None, timeout=100, verbose:bool = False, num_trials=5, **kwargs):

        output = {}
        host_map = cls.hosts(search=search)
        if hosts != None:
            if isinstance(hosts, str):
                host_map = {k:v for k,v in host_map.items() if hosts in k}
            elif isinstance(hosts, list):
                host_map = {k:v for k,v in host_map.items() if k in hosts}
            else:
                raise Exception(f'hosts must be a list or a string')
        hosts = host_map
        for i in range(num_trials):
            try:
                results = {}
                for host in host_map:
                    result_future = c.submit(cls.ssh_cmd, args=commands, kwargs=dict(host=host, cwd=cwd, verbose=verbose,**kwargs), return_future=True)
                    results[host] = result_future
                result_values = c.wait(list(results.values()), timeout=timeout)
                results =  dict(zip(results.keys(), result_values))
                results =  {k:v for k,v in results.items()}

                if all([v == None for v in results.values()]):
                    raise Exception(f'all results are None')
                
                unfinished_hosts  = []
                for k, v in results.items():
                    if v == None:
                        unfinished_hosts += [k]
                    else:
                        output[k] = v

                host_map = {k:v for k,v in host_map.items() if k in unfinished_hosts}
                
                if len(host_map) == 0:
                    break

            except Exception as e:
                c.print('Retrying')
                c.print(c.detailed_error(e))
                continue




        return output 

    
    @classmethod
    def add_admin(cls):
        root_key_address = c.root_key().ss58_address
        return cls.cmd(f'c add_admin {root_key_address}')
    
    @classmethod
    def is_admin(cls):
        root_key_address = c.root_key().ss58_address
        results =  cls.cmd(f'c is_admin {root_key_address}')
        for host, r in results.items():
            results[host] = bool(r)
        return results
    
    @classmethod
    def add_servers(cls, *args, add_admins:bool=False, network='remote'):
    

        if add_admins:
            c.print('Adding admin')
            cls.add_admin()
        servers = list(cls.cmd('c addy', verbose=True).values())
        for i, server in enumerate(servers):
            if server.endswith('\n'):
                servers[i] = server[:-1]
        c.add_servers(*servers, network=network)
        cls.check_servers()
        servers = c.servers(network=network)
        return {'status': 'success', 'msg': f'Servers added', 'servers': servers}

    @classmethod
    def servers(self, network='remote'):
        return c.servers(network=network)
    
    @classmethod
    def namespace(self, network='remote'):
        return c.namespace(network=network)

    @classmethod
    def get_address(self, name):
        return c.get_address(name)

    
    @classmethod
    def addresses(self, network='remote'):
        return c.addresses(network=network)
    
    @classmethod
    def servers_info(self, network='remote'):
        return c.servers_info(network=network)
    @classmethod
    def push(cls,**kwargs):
        return [c.push(), cls.pull()]
        
    @classmethod
    def pull(cls, stash=True):
        return c.rcmd(f'c pull stash={stash}')
    @classmethod
    def check_servers(cls):
        for m,a in c.namespace(network='remote').items():
            try:
                result = c.call(a)
                c.print(f'{c.emoji("checkmaark")} [bold green]{m} --> {a}[/bold green] {c.emoji("checkmark")}')
            except Exception as e:
                c.rm_server(a, network='remote')
                c.print('failed')


    

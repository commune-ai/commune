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
                c.print(f'[bold]{host}[/bold]', line.strip('\n'))
            outputs['error'] += line


        if len(outputs['error']) == 0:
            output = outputs['output']
        
        stdin.close()
        stdout.close()
        stderr.close()
        # Close the Remote connection
        client.close()

        return output


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
    def host(self, name):
        hosts = self.hosts()

        if name not in hosts:
            raise Exception(f'Host {name} not found')
        
        return hosts[name]
    

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
    def cmd(cls, *commands,  search=None, cwd=None, timeout=100, return_list:bool = False, stream=True, **kwargs):
        hosts = cls.hosts(search=search)
        results = {}
        for host in hosts:
            result_future = c.submit(cls.ssh_cmd, args=commands, kwargs=dict(host=host, cwd=cwd, stream=stream, **kwargs), return_future=True, timeout=timeout)
            results[host] = result_future

        result_values = c.wait(list(results.values()), timeout=timeout)
        results =  dict(zip(results.keys(), result_values))
        results =  {k:v for k,v in results.items()}
        for k,v in results.items():
            if isinstance(v, str):
                if  v.endswith('\n'):
                    results[k] =  v[:-1]
        if return_list:
            return list(results.values())

        return results
    
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
        c.add_servers(*servers, network=network)

    @classmethod
    def servers(self, network='remote'):
        return c.servers(network=network)
    
    @classmethod
    def addresses(self, network='remote'):
        return c.addresses(network=network)
    
    @classmethod
    def servers_info(self, network='remote'):
        return c.servers_info(network=network)
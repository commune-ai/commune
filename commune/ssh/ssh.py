import commune as c

class SSH(c.Module):
    host_data_path = f'{c.datapath}/ssh.json'
    @classmethod
    def call(cls, 
            *cmd_args, host  = None,  cwd=None, verbose=False, **kwargs ):
        """s
        Run a command on a remote server using SSH.

        :param host: Hostname or IP address of the remote machine.
        :param port: SSH port (typically 22).
        :param username: SSH username.
        :param password: SSH password.
        :param command: Command to be executed on the remote machine.
        :return: Command output.
        """
        command = ' '.join(cmd_args)

        if cwd != None:
            command = f'cd {cwd} ; {command}'


        import paramiko
        hosts = cls.hosts()
        if host == None:
            host = list(hosts.keys())[0]
        if host not in hosts:
            raise Exception(f'Host {host} not found')
        host = hosts[host]

        # Create an SSH client instance.
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

        output = []
        for line in stdout.readlines():
            output += [line.strip()]
            if verbose:
                text = f'[green]{host}[/green] | {line.strip()}'
                c.print(text, color='green')


        error = []
        for line in stderr.readlines():
            error += [line.strip()]
            if verbose:
                text = f'[red]{host}[/red] | {line.strip()}'
                c.print(text, color='red')
  
        # Return the result
        error = '\n'.join(error)
        output = '\n'.join(output)

        if len(error) > 0:
            output = {'error': error, 'output': output}
        
        stdin.close()
        stdout.close()
        stderr.close()
        # Close the SSH connection
        client.close()

        return output

    def serve(self, **kwargs):
        return self.call(**kwargs)
    
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
        # Test SSH
        c.print(self.call('ls'))

    @classmethod
    def pool(cls, *commands,  search=None, cwd=None, timeout=100, **kwargs):
        hosts = cls.hosts(search=search)
        results = {}
        for host in hosts:
            result_future = c.submit(cls.call, args=commands, kwargs=dict(host=host, cwd=cwd, **kwargs), return_future=True)
            results[host] = result_future

        result_values = c.wait(list(results.values()), timeout=timeout)
        results =  dict(zip(results.keys(), result_values))
        results =  {k:v for k,v in results.items()}
        for k,v in results.items():
            if isinstance(v, str):
                results[k] = v.split('\n')
                if len(results[k])>0 and len(results[k][-1]) == 0:
                    results[k] = results[k][:-1]
            


        return results
    

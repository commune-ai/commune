import commune as c

class SSH(c.Module):
    host_data_path = f'{c.datapath}/ssh.json'
    @classmethod
    def call(cls, 
            *cmd_args, hostname = None,  cwd=None, stream=False, **kwargs ):
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
        import paramiko

        hosts = cls.hosts()
        if hostname == None:
            hostname = list(hosts.keys())[0]
        if hostname not in hosts:
            raise Exception(f'Host {host} not found')
        
        host = hosts[hostname]
        hostname = host['host']
        port = host['port']
        user = host['user']
        pwd = host['pwd']
        if cwd != None:
            command = f'cd {cwd} ; {command}'

        # Create an SSH client instance.
        client = paramiko.SSHClient()
        
        # Automatically add the server's host key (this is insecure and used for demonstration; 
        # in production, you should have the remote server's public key in known_hosts)
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the remote server
        client.connect(hostname, port=port, username=user, password=pwd)
        
        # Execute command
        stdin, stdout, stderr = client.exec_command(command)


        if stream:
            # Print the output of ls command
            def generate_output():
                for line in stdout.readlines():
                    yield line.strip('\n')
                
            return generate_output()

        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

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
                 name = None
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
            name = f'server{cnt}'

            while name in hosts:
                name = f'server{cnt}'
                cnt += 1
        
        hosts[name] = host
        cls.put_json(cls.host_data_path, hosts)
        return {'status': 'success', '': f'Host added', }
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
        return self.hosts()[name]
    
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
    def callpool(cls, *commands,  search=None, cwd=None, **kwargs):
        hosts = cls.hosts(search=search)
        results = {}
        for host in hosts:
            result_future = c.submit(cls.call, args=commands, kwargs=dict(hostname=host, cwd=cwd, **kwargs), return_future=True)
            results[host] = result_future

        result_values = c.wait(list(results.values()))
        results =  dict(zip(results.keys(), result_values))
        return results

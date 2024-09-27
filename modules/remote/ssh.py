import commune as c

import commune as c
import streamlit as st
from typing import *
import paramiko

class SSH(c.Module):
    def __init__(self):
        pass
    def forward(self, *cmd_args, 
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

        c.print(f'Running command: {command} on {host["name"]}')

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
    
            # stdin.close()
            # stdout.close()
            # stderr.close()
            # client.close()
        except Exception as e:
            c.print(e)

        return output

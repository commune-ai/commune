import commune as c
import streamlit as st
from typing import *
import json
import paramiko
import os

class SSH(c.Module):
    @classmethod
    def cmd(cls, *cmd_args, 
                host:str= None,  
                cwd:str=None, 
                verbose=False, 
                sudo=False, 
                key=None, 
                timeout=10,  
                generator=True,
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
        
        if command.startswith('c '):
            command = command.replace('c ', cls.executable_path + ' ')

        if cwd != None:
            command = f'cd {cwd} && {command}'

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

            if generator:
                for line in stdout.readlines():
                    if verbose:
                        c.print(f'[bold]{host["host"]}[/bold]', line.strip('\n'), color=color)
                    yield line

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
    
            # stdin.close()
            # stdout.close()
            # stderr.close()
            # client.close()
        except Exception as e:
            c.print(e)
        return outputs


    key_path = os.path.expanduser('~/.ssh/id_ed25519')
    public_key_path = key_path + '.pub'


    def public_key_text(self):
        return c.get_text(self.public_key_path)


    def public_key(self):
        return self.public_key_text().split('\n')[0].split(' ')[1]

    def create_key(self, key:str = None):
        if key == None:
            key = self.key_path
        return c.cmd(f'ssh-keygen -o -a 100 -t ed25519 -f ~/.ssh/id_ed25519')
    

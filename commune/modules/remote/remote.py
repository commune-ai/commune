import commune as c
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

        return {'status': 'success', 'msg': f'Hosts saved', 'hosts': hosts, 'path': cls.host_data_path, 'filetype': filetype}
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
    def host2ip(cls):
        hosts = cls.hosts()
        return {k:v['host'] for k,v in hosts.items()}

    @classmethod
    def ip2host(cls):
        host2ip = cls.host2ip()
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
    
    @classmethod
    def add_servers(cls, *args, add_admins:bool=False, timeout=20, refresh=False, network='remote'):
        if refresh:
            c.rm_namespace(network=network)
        server_addresses = list(cls.cmd('c addy', verbose=True, timeout=timeout).values())
        servers = c.servers(network=network)
        for i, server_address in enumerate(server_addresses):
            if isinstance(server_address, str):
                if server_address.endswith('\n'):
                    server_address = server_address[:-1]
                servers += [server_address]
        c.add_servers(*servers, network=network)
        servers = c.servers(network=network)
        return {'status': 'success', 'msg': f'Servers added', 'servers': servers}

    @classmethod
    def servers(self,search: str ='module', network='remote'):
        return c.servers(search, network=network)
    
    @classmethod
    def serve(cls, *args, n=1, **kwargs):
        return cls.call('serve', *args, search='module', n=n, **kwargs)
    @classmethod
    def fleet(cls, module, tag='', n=1, timeout=100, **kwargs):

        futures = []

        for i in range(n):
            f = cls.call('serve', f'{module}::{tag}{i}', return_future=True, n=1, **kwargs)
            c.print(f)
            futures += [f]
        return c.wait(futures, timeout=timeout)

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
    def namespace(cls, search='module', network='remote', update=True):

        if update:
            namespace = {}
            host2namespace = cls.call('namespace', public=True, search=search, timeout=20)
            for host, host_namespace in host2namespace.items():
                for name, address in host_namespace.items():
                    tag = ''
                    while name + str(tag) in namespace:
                        if tag == '':
                            tag = 1
                        else:
                            tag += 1
                    namespace[name + str(tag)] = address
            c.print(namespace)
            c.put_namespace(namespace=namespace, network=network)
        else:
            namespace = c.get_namespace(search, network=network)

        return namespace

    @classmethod
    def get_address(self, name):
        return c.get_address(name)

    
    @classmethod
    def addresses(self, network='remote'):
        return c.addresses(network=network)
    
    @classmethod
    def server_infos(self, network='remote'):
        return c.server_infos(network=network)
    @classmethod
    def push(cls,**kwargs):
        return [c.push(), cls.pull()]
    

    @classmethod
    def call(cls, fn:str='info' , *args, search:str='module',  network:str='remote', n:int=None, return_future: bool = False, timeout=20, **kwargs):
        futures = {}
        kwargs['network'] =  network
        namespace = c.namespace(search=search, network=network)
        if n == None:
            n = len(namespace)
        for name, address in c.shuffle(list(namespace.items()))[:n]:
            futures[name] = c.submit(c.call, args=(address, fn, *args), kwargs=kwargs, return_future=True, timeout=timeout)
        
        if return_future:
            if len(futures) == 1:
                return list(futures.values())[0]
            return futures
        else:

            results = c.wait(list(futures.values()), timeout=timeout)
            results = dict(zip(futures.keys(), results))
            if len(results) == 1:
                return list(results.values())[0]
        
            return results

        
        
    @classmethod
    def pull(cls, stash=True, hosts=None):
        return c.rcmd(f'c pull stash={stash}', hosts=hosts)
    
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
                ssh_login_param = f'ssh {host["user"]}@{host["host"]} -p {host["port"]}'
                cols[1].code(ssh_login_param)
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

        try:
            for result in c.wait(futures, timeout=timeout, generator=True, return_dict=True):
                host = hosts[result['idx']]
                if host == None:
                    continue
                host2future.pop(host)

                result = result['result']
                if c.is_error(result):
                    with st.expander(host + ' ' +  c.emoji('cross'), expanded=True):
                        st.markdown(f"""```bash
                                    {result}```""")
                else:
                    with st.expander(host + ' ' +  c.emoji('checkmark'), expanded=True):
                        st.markdown(f"""```bash
                                    {result}```""")
        except Exception as e:
            pending_hosts = list(host2future.keys())
            st.error(c.detailed_error(e))
            st.error(f"Hosts {pending_hosts} timed out")

        



       

    dash = dashboard




    # @classmethod
    # def refresh_servers(cls):
    #     cls.cmd('')
    
Remote.run(__name__)

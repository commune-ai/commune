import commune as c
from typing import *
import os

# A NETWORK IS A 
class Network(c.Module):

    # the default
    network : str = 'local'

    def __init__(self, network:str=None, **kwargs):
        self.set_network(network)

    def set_network(self, network:str):
        self.network = network or self.network
        return self.network
    
    def networks(self, module_prefix:str='network') -> List[str]:
        networks = []
        for m in c.modules(module_prefix):
            if not m.startswith(module_prefix):
                continue
            if m.count('.') == 1:
                network = m.split('.')[-1]
            elif m == module_prefix:
                network = 'local'
            else:
                continue
            networks.append(network)
        networks = sorted(list(set(networks)))
        return networks
    
    def namespace(self, 
                  network:str = 'local', 
                  search=None, 
                  max_age:int = 60,
                  update:bool = False,
                  timeout=6) -> dict:
        network = self.resolve_network(network)
        path = self.resolve_network_path(network)
        namespace = self.get(path, None, max_age=max_age, update=update)
        if namespace == None:
            self.put(path,namespace)
            if 'local' == network: 
                namespace = self.update_local_namespace(timeout=timeout)
            elif c.module_exists(network):
                network_module = c.module(network)()
                namespace =  network_module.namespace(search=search, update=True)
            else: 
                namespace = {}
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k} 
        namespace = dict(sorted(namespace.items(), key=lambda x: x[0]))
        if network == 'local':
            loopback = '0.0.0.0'
            namespace = {k: loopback + ':' + v.split(':')[-1] for k,v in namespace.items() }
        return namespace
    
    def update_local_namespace(self, timeout=2):
        namespace = {}
        addresses = ['0.0.0.0'+':'+str(p) for p in c.used_ports()]
        future2address = {}
        for address in addresses:
            f = c.submit(c.call, [address+'/server_name'], timeout=timeout)
            future2address[f] = address
        futures = list(future2address.keys())
        try:
            for f in c.as_completed(futures, timeout=timeout):
                try:
                    address = future2address[f]
                    name = f.result()
                    if isinstance(name, str):
                        namespace[name] = address
                except Exception as e:
                    print('Error in ', e)
        except Exception as e:
            print('Error in ', e)
        namespace = {k:v for k,v in namespace.items() if 'Error' not in k} 
        namespace = {k: '0.0.0.0:' + str(v.split(':')[-1]) for k,v in namespace.items() }
        return namespace 
    
    get_namespace = _namespace = namespace

    
    def register_server(self, name:str, address:str, network=network) -> None:
        namespace = self.namespace(network=network)
        namespace[name] = address
        self.put_namespace(network, namespace)
        return {'success': True, 'msg': f'Block {name} registered to {network}.'}
    
    
    def deregister_server(self, name:str, network=network) -> Dict:
        namespace = self.namespace(network=network)
        address2name = {v: k for k, v in namespace.items()}
        if name in address2name:
            name = address2name[name]
        if name in namespace:
            del namespace[name]
            self.put_namespace(network, namespace)
            return {'status': 'success', 'msg': f'Block {name} deregistered.'}
        else:
            return {'success': False, 'msg': f'Block {name} not found.'}
    
    
    def rm_server(self,  name:str, network=network):
        return self.deregister_server(name, network=network)
    
    
    def get_address(self, name:str, network:str=network, external:bool = True) -> dict:
        namespace = self.namespace(network=network)
        address = namespace.get(name, None)
        if external and address != None:
            address = address.replace(c.default_ip, c.ip()) 
        return address

    
    def put_namespace(self, network:str, namespace:dict) -> None:
        assert isinstance(namespace, dict), 'Network must be a dict.'
        return self.put(network, namespace)        
    
    add_namespace = put_namespace
    
    
    def rm_namespace(self,network:str) -> None:
        if self.namespace_exists(network):
            self.rm(network)
            return {'success': True, 'msg': f'Network {network} removed.'}
        else:
            return {'success': False, 'msg': f'Network {network} not found.'}
        
    def resolve_network(self, network:str) -> str:
        return network or self.network
    
    def resolve_network_path(self, network:str) -> str:
        return self.resolve_path(self.resolve_network(network))
    def namespace_exists(self, network:str) -> bool:
        path = self.resolve_network_path( network)
        return os.path.exists(path)
    
    
    def modules(self, network:List=network) -> List[str]:
        return list(self.namespace(network=network).keys())
    
    
    def addresses(self, network:str=network, **kwargs) -> List[str]:
        return list(self.namespace(network=network, **kwargs).values())
    
    
    def check_servers(self, *args, **kwargs):
        servers = c.pm2ls()
        namespace = c.get_namespace(*args, **kwargs)
        c.print('Checking servers', color='blue')
        for server in servers:
            if server in namespace:
                c.print(c.pm2_restart(server))

        return {'success': True, 'msg': 'Servers checked.'}

    
    def add_server(self, address:str, network:str = 'local', name=None,timeout:int=4, **kwargs):
        """
        Add a server to the namespace.
        """
        module = c.connect(address)
        info = module.info(timeout=timeout)
        name = info['name'] if name == None else name
        # check if name exists
        address = info['address']
        module_ip = address.split(':')[0]
        is_remote = bool(module_ip != c.ip())
        namespace = self.namespace(network=network)
        if is_remote:
            name = name + '_' + str(module_ip)
        addresses = list(namespace.values())
        if address not in addresses:
            return {'success': False, 'msg': f'{address} not in {addresses}'}
        namespace[name] = address
        self.put_namespace(network, namespace)

        return {'success': True, 'msg': f'Added {address} to {network} modules', 'remote_modules': self.servers(network=network), 'network': network}
    
    
    def rm_server(self,  name, network:str = 'local', **kwargs):
        namespace = self.namespace(network=network)
        if name in namespace.values():
            for k, v in c.copy(list(namespace.items())):
                if v == name:
                    name = k
                    break
        if name in namespace:
            # reregister
            address = self.get_address(name, network=network)
            self.deregister_server(name, network=network)
            servers = self.servers(network=network)
            assert self.server_exists(name, network=network) == False, f'{name} still exists'
            return {'success': True, 'msg': f'removed {address} to remote modules', 'servers': servers, 'network': network}
        else:
            return {'success': False, 'msg': f'{name} does not exist'}

    
    def servers(self, search=None, network:str = 'local',  **kwargs):
        namespace = self.namespace(search=search, network=network, **kwargs)
        return list(namespace.keys())
    
    
    def server_exists(self, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = self.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists
    
    
    def server_exists(self, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = self.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists  
    
Network.run(__name__)



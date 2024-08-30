import commune as c
from typing import *
import os

# THIS IS WHAT THE INTERNET IS, A BUNCH OF NAMESPACES, AND A BUNCH OF SERVERS, AND A BUNCH OF MODULES.
# THIS IS THE INTERNET OF INTERNETS.
class Namespace(c.Module):

    # the default
    network : str = 'local'
    @classmethod
    def resolve_network_path(cls, network:str, netuid:str=None):
        if netuid != None:
            if network not in ['subspace']:
                network = f'subspace'
            network = f'subspace/{netuid}'
        return cls.resolve_path(network + '.json')

    @classmethod
    def namespace(cls, search=None,
                    network:str = 'local',
                    update:bool = False, 
                    netuid=None, 
                    max_age:int = 60,
                    timeout=6,
                    verbose=False) -> dict:
        network = network or 'local'
        path = cls.resolve_network_path(network)
        namespace = cls.get(path, None, max_age=max_age)
        if namespace == None:
            namespace = cls.update_namespace(network=network, 
                                            netuid=netuid, 
                                            timeout=timeout, 
                                            verbose=verbose)
            cls.put(path,namespace)
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k} 

        namespace = cls.clean_namespace(namespace)

        return namespace
    
    @classmethod
    def clean_namespace(cls, namespace):
        namespace = {k:':'.join(v.split(':')[:-1]) + ':'+ str(v.split(':')[-1]) for k,v in namespace.items()}
        namespace = dict(sorted(namespace.items(), key=lambda x: x[0]))
        ip  = c.ip()
        namespace = {k: v.replace(ip, '0.0.0.0') for k,v in namespace.items() }
        namespace = { k.replace('"', ''): v for k,v in namespace.items() }
        return namespace
    @classmethod
    def update_namespace(cls, network, netuid=None, timeout=5, search=None, verbose=False):
        c.print(f'UPDATING --> NETWORK(network={network} netuid={netuid})', color='blue')

        if 'subspace' in network:
            if '.' in network:
                network, netuid = network.split('.')
            else: 
                netuid = netuid or 0
            if c.is_int(netuid):
                netuid = int(netuid)
            namespace = c.module(network)().namespace(search=search, max_age=1, netuid=netuid)
            return namespace
        elif 'local' == network: 
            namespace = {}
            addresses = ['0.0.0.0'+':'+str(p) for p in c.used_ports()]
            future2address = {}
            for address in addresses:
                f = c.submit(c.call, [address+'/server_name'], timeout=timeout)
                future2address[f] = address
            futures = list(future2address.keys())
            try:
                for f in c.as_completed(futures, timeout=timeout):
                    address = future2address[f]
                    try:
                        name = f.result()
                        namespace[name] = address
                    except Exception as e:
                        c.print(f'Error {e} with {name} and {address}', color='red', verbose=True)
            except Exception as e:
                c.print(f'Error: {e}', color='red', verbose=True) 
            namespace = {k:v for k,v in namespace.items() if 'Error' not in k} 
            ip  = c.ip(update=1)
            namespace = {k: v.replace(ip, '0.0.0.0') for k,v in namespace.items() }
        else:
            return {}
        return namespace 
    
    get_namespace = _namespace = namespace

    @classmethod
    def register_server(cls, name:str, address:str, network=network) -> None:
        namespace = cls.namespace(network=network)
        namespace[name] = address
        cls.put_namespace(network, namespace)
        return {'success': True, 'msg': f'Block {name} registered to {network}.'}
    
    @classmethod
    def deregister_server(cls, name:str, network=network) -> Dict:
        namespace = cls.namespace(network=network)
        address2name = {v: k for k, v in namespace.items()}
        if name in address2name:
            name = address2name[name]
        if name in namespace:
            del namespace[name]
            cls.put_namespace(network, namespace)
            return {'status': 'success', 'msg': f'Block {name} deregistered.'}
        else:
            return {'success': False, 'msg': f'Block {name} not found.'}
    
    @classmethod
    def rm_server(self,  name:str, network=network):
        return self.deregister_server(name, network=network)
    
    @classmethod
    def get_address(cls, name:str, network:str=network, external:bool = True) -> dict:
        namespace = cls.namespace(network=network)
        address = namespace.get(name, None)
        if external and address != None:
            address = address.replace(c.default_ip, c.ip()) 
        return address

    @classmethod
    def put_namespace(cls, network:str, namespace:dict) -> None:
        assert isinstance(namespace, dict), 'Namespace must be a dict.'
        return cls.put(network, namespace)        
    
    add_namespace = put_namespace
    
    @classmethod
    def rm_namespace(cls,network:str) -> None:
        if cls.namespace_exists(network):
            cls.rm(network)
            return {'success': True, 'msg': f'Namespace {network} removed.'}
        else:
            return {'success': False, 'msg': f'Namespace {network} not found.'}
        
    @classmethod
    def networks(cls) -> dict:
        return [p.split('/')[-1].split('.')[0] for p in cls.ls()]
    
    @classmethod
    def namespace_exists(cls, network:str) -> bool:
        path = cls.resolve_network_path( network)
        return os.path.exists(path)
    
    @classmethod
    def modules(cls, network:List=network) -> List[str]:
        return list(cls.namespace(network=network).keys())
    
    @classmethod
    def addresses(cls, network:str=network, **kwargs) -> List[str]:
        return list(cls.namespace(network=network, **kwargs).values())
    
    @classmethod
    def module_exists(cls, module:str, network:str=network) -> bool:
        namespace = cls.namespace(network=network)
        return bool(module in namespace)

    @classmethod
    def check_servers(self, *args, **kwargs):
        servers = c.pm2ls()
        namespace = c.namespace(*args, **kwargs)
        c.print('Checking servers', color='blue')
        for server in servers:
            if server in namespace:
                c.print(c.pm2_restart(server))

        return {'success': True, 'msg': 'Servers checked.'}

    @classmethod
    def add_server(cls, address:str, name=None, network:str = 'local',timeout:int=4, **kwargs):
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
        namespace = cls.namespace(network=network)
        if is_remote:
            name = name + '_' + str(module_ip)
        addresses = list(namespace.values())
        if address not in addresses:
            return {'success': False, 'msg': f'{address} not in {addresses}'}
        namespace[name] = address
        cls.put_namespace(network, namespace)

        return {'success': True, 'msg': f'Added {address} to {network} modules', 'remote_modules': cls.servers(network=network), 'network': network}
    
    @classmethod
    def remote_servers(cls, network:str = 'remote', **kwargs):
        return cls.namespace(network=network)
    
    @classmethod
    def infos(cls, search=None, network=network, servers=None, features=['key', 'address', 'name'],   update:str=True, batch_size = 10, timeout=20, hardware=True, namespace=True, schema=True) -> List[str]:
        path = f'infos/{network}'
        if not update:
            infos = cls.get(path, [])
        if update or len(infos) == 0:
            infos = []
            servers = cls.servers(search=search, network=network) if servers == None else servers
            futures = []
            infos = []
            for s in servers:
                kwargs = {'module':s, 'fn':'info', 'network': network, 'hardware': hardware, 'namespace': namespace, 'schema': schema}
                future = c.submit(c.call, kwargs=kwargs, return_future=True, timeout=timeout)
                futures.append(future)
                if len(futures) >= batch_size:
                    for f in c.as_completed(futures):
                        
                        result = f.result()
                        futures.remove(f)

                        if isinstance(result, dict) and 'error' not in result:
                            infos.append(result)
                        
                        break
            infos += c.wait(futures, timeout=timeout)
            cls.put(path, infos)
        infos = [s for s in infos if s != None]
        if search != None:
            infos = [s for s in infos if 'name' in s and search in s['name']]

        if features != None:
            infos = [{k:v for k,v in s.items() if k in features} for s in infos]
        return infos
    
    @classmethod
    def rm_server(cls,  name, network:str = 'local', **kwargs):
        namespace = cls.namespace(network=network)
        if name in namespace.values():
            for k, v in c.copy(list(namespace.items())):
                if v == name:
                    name = k
                    break
            
        if name in namespace:
            # reregister
            address = cls.get_address(name, network=network)
            cls.deregister_server(name, network=network)
            servers = cls.servers(network=network)
            assert cls.server_exists(name, network=network) == False, f'{name} still exists'
            return {'success': True, 'msg': f'removed {address} to remote modules', 'servers': servers, 'network': network}
        else:
            return {'success': False, 'msg': f'{name} does not exist'}

    @classmethod
    def servers(cls, search=None, network:str = 'local',  **kwargs):
        namespace = cls.namespace(search=search, network=network, **kwargs)
        return list(namespace.keys())
    
    @classmethod
    def refresh_namespace(cls, network:str):
        return cls.put_namespace(network, {})
    @classmethod
    def network2namespace(self):
        return {network: self.namespace(network=network) for network in self.networks()}
    all = network2namespace
    @classmethod
    def server_exists(cls, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = cls.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists
    
    @classmethod
    def test(cls):
        network = 'test_namespace'
        cls.rm_namespace(network)
        namespace = cls.namespace(network=network)
        assert cls.namespace(network=network) == {}, f'Namespace not empty., {namespace}'
        name = 'test'
        address =  '0.0.0.0:8888'
        cls.register_server(name=name, address=address, network=network)
        namespace = cls.namespace(network=network)
        assert  namespace[name] == address, f'Namespace not updated. {namespace}'
        cls.deregister_server(name, network=network)
        assert cls.namespace(network=network) == {}
        cls.rm_namespace(network)
        assert cls.namespace_exists(network) == False        
        return {'success': True, 'msg': 'Namespace tests passed.'}
    

    
    @classmethod
    def server_exists(cls, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = cls.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists

    @classmethod
    def clean(cls, network='local'):
        namespace = cls.namespace(network=network)
        address2name = {}
        
        for name, address in address2name.items():
            if address in address2name.values():
                if len(address2name[address]) < len(name):
                    namespace[address] = name
            else:
                address2name[address] = name

        namespace = {v:k for k,v in address2name.items()}
        return namespace     
    
Namespace.run(__name__)



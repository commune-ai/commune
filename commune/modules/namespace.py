import commune as c
from typing import *

# THIS IS WHAT THE INTERNET IS, A BUNCH OF NAMESPACES, AND A BUNCH OF SERVERS, AND A BUNCH OF MODULES.
# THIS IS THE INTERNET OF INTERNETS.
class Namespace(c.Module):

    # the default
    network : str = 'local'

    @classmethod
    def register_server(cls, name:str, address:str, network=network) -> None:
        namespace = cls.get_namespace(network=network)
        namespace[name] = address
        cls.put_namespace(network, namespace)
        return {'success': True, 'msg': f'Block {name} registered to {network}.'}

    @classmethod
    def deregister_server(cls, name:str, network=network) -> Dict:

        namespace = cls.get_namespace(network=network)
        if name in namespace:
            del namespace[name]
            cls.put_namespace(network, namespace)
            return {'status': 'success', 'msg': f'Block {name} deregistered.'}
        else:
            return {'success': False, 'msg': f'Block {name} not found.'}

    @classmethod
    def get_address(cls, name:str, network:str=network, external:bool = True) -> dict:
        namespace = cls.get_namespace(network=network)
        address = namespace.get(name, None)
        if external and address != None:
            address = address.replace(c.default_ip, c.ip())
            
        return address

    @classmethod
    def get_namespace(cls, search=None, network:str = 'local', update:bool = False, public:bool = False) -> dict:
        if network == None: 
            network = cls.network

        if network == 'subspace':
            namespace =  c.module(network)().namespace(update=update)
        else:
            if update:
                cls.update_namespace(network=network, full_scan=bool(network=='local'))
            namespace = cls.get(network, {})
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k}

        if public:
            ip = c.ip()
            namespace = {k:v.replace(c.default_ip, ip) for k,v in namespace.items()}
        return namespace
    
    
    @classmethod
    def put_namespace(cls, network:str, namespace:dict) -> None:
        assert isinstance(namespace, dict), 'Namespace must be a dict.'
        cls.put(network, namespace)
        return {'success': False, 'msg': f'Namespace {network} updated.'}

    

    @classmethod
    def rm_namespace(cls,network:str) -> None:
        if cls.exists(network):
            cls.rm(network)
            return {'success': True, 'msg': f'Namespace {network} removed.'}
        else:
            return {'success': False, 'msg': f'Namespace {network} not found.'}
    @classmethod
    def name2address(cls, name:str, network:str=network ):
        namespace = cls.get_namespace(network=network)
        address =  namespace.get(name, None)
        ip = c.ip()
    
        address = address.replace(c.default_ip, ip)
        assert ip in address, f'ip {ip} not in address {address}'
        return address
    @classmethod
    def networks(cls) -> dict:
        return [p.split('/')[-1].split('.')[0] for p in cls.ls()]
    
    @classmethod
    def namespace_exists(cls, network:str) -> bool:
        return cls.exists(network)


    @classmethod
    def modules(cls, network:List=network) -> List[str]:
        return list(cls.get_namespace(network=network).keys())
    
    @classmethod
    def addresses(cls, network:str=network, **kwargs) -> List[str]:
        return list(cls.get_namespace(network=network, **kwargs).values())
    
    @classmethod
    def module_exists(cls, module:str, network:str=network) -> bool:
        namespace = cls.get_namespace(network=network)
        return bool(module in namespace)

    @classmethod
    def update_namespace(cls,
                        chunk_size:int=10, 
                        timeout:int = 10,
                        full_scan:bool = True,
                        network:str = network,)-> dict:
        '''
        The module port is where modules can connect with each othe.
        When a module is served "module.serve())"
        it will register itself with the namespace_local dictionary.
        '''

        namespace = cls.get_namespace(network=network, update=False) # get local namespace from redis
        addresses = c.copy(list(namespace.values()))
        namespace = {}

        if full_scan == True or len(addresses) == 0:
            addresses = [c.default_ip+':'+str(p) for p in c.used_ports()]


        for i in range(0, len(addresses), chunk_size):
            addresses_chunk = addresses[i:i+chunk_size]
            names_chunk = c.gather([c.async_call(address, fn='server_name', timeout=timeout) for address in addresses_chunk])
            for i in range(len(names_chunk)):
                if isinstance(names_chunk[i], str):
                    namespace[names_chunk[i]] = addresses_chunk[i]

        cls.put_namespace(network, namespace)
            
        return namespace
    
    @classmethod
    def migrate_namespace(cls, network:str='local'):
        namespace = c.get_json('local_namespace', {})
        c.put_namespace(network, namespace)

    @classmethod
    def merge_namespace(cls, from_network:str, to_network:str, module = None):
        from_namespace = c.get_namespace(network=from_network)
        if module == None:
            module = c.module(from_network)

        to_namespace = c.get_namespace(network=to_network)
        to_namespace.update(from_namespace)
        c.put_namespace(to_network, to_namespace)
        return {'success': True, 'msg': f'Namespace {from_network} merged into {to_network}.'}


    remote_modules_path ='remote_modules'
    @classmethod
    def add_server(cls, address:str, name=None, network:str = 'local', **kwargs):
        module = c.connect(address)
        module_info = module.info()
        name = module_info['name'] if name == None else name
        # check if name exists
        namespace = cls.get_namespace(network=network)
        base_name = c.copy(name)
        cnt = 0
        if address in namespace.values():
            for k, v in c.copy(list(namespace.items())):
                if v == address:
                    namespace.pop(k)

        while name in namespace:
            name = base_name +address[:3 + cnt].replace('.', '')
            cnt += 1

        namespace[name] = address
        cls.put_namespace(network, namespace)

        return {'success': True, 'msg': f'Added {address} to {network} modules', 'remote_modules': cls.servers(network=network), 'network': network}
    
    

    @classmethod
    def remote_servers(cls, network:str = 'local', **kwargs):
        return c.namespace(network=network)
    
    @classmethod
    def add_servers(cls, *servers, network:str='local', **kwargs):
        responses = []
        for server in servers:
            try:
                response = cls.add_server(server, network=network)
            except Exception as e:
                response = {'success': False, 'msg': str(e)}
            responses.append(response)

        return responses


    
    @classmethod
    def servers_info(cls, search=None, network=network, update:str=False, batch_size = 10, timeout=4) -> List[str]:
        if not update:
            servers_info = cls.get('servers_info', [])

        if update or len(servers_info) == 0:
            servers_info = []
            servers = cls.servers(search=search, network=network)
            futures = []
            servers_info = []
            for s in servers:
                kwargs = {'module':s, 'fn':'info', 'network': network}
                future = c.submit(c.call, kwargs=kwargs, return_future=True, timeout=timeout)
                futures.append(future)
                if len(futures) >= batch_size:
                    for f in c.as_completed(futures):
                        servers_info.append(f.result())
                        futures.remove(f)
                        break
            servers_info += c.wait(futures, timeout=timeout)
            cls.put('servers_info', servers_info)
        return [s for s in servers_info if s != None]
    
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
            remote_modules = c.get(cls.remote_modules_path, {})
            remote_modules.pop(name, None)
            servers = cls.servers(network=network)
            assert cls.server_exists(name, network=network) == False, f'{name} still exists'
            return {'success': True, 'msg': f'removed {address} to remote modules', 'servers': servers, 'network': network}
        else:
            return {'success': False, 'msg': f'{name} does not exist'}
        

    @classmethod
    def namespace(cls, search=None, network:str = 'local', **kwargs):
        namespace = cls.get_namespace(network=network, **kwargs)
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k}

        return namespace

    

    @classmethod
    def servers(cls, search=None, network:str = 'local', **kwargs):
        namespace = cls.namespace(search=search, network=network)
        return list(namespace.keys())


    @classmethod
    def has_server(cls, name:str, network:str = 'local', **kwargs):
        return cls.server_exists(name, network=network)
    
    @classmethod
    def refresh_namespace(cls, network:str):
        return cls.put_namespace(network, {})
    @classmethod
    def network2namespace(self):
        return {network: self.get_namespace(network=network) for network in self.networks()}
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
        network = 'test'
        network2  = 'test2'
        cls.rm_namespace(network)
        cls.rm_namespace(network2)

        assert cls.get_namespace(network=network) == {}, 'Namespace not empty.'
        cls.register_server('test', 'test', network=network)
        assert cls.get_namespace(network=network) == {'test': 'test'}, f'Namespace not updated. {cls.get_namespace(network=network)}'

        assert cls.get_namespace(network2) == {}
        cls.register_server('test', 'test', network=network2)
        assert cls.get_namespace(network=network) == {'test': 'test'}, f'Namespace not restored. {cls.get_namespace(network=network)}'
        cls.deregister_server('test', network=network2)
        assert cls.get_namespace(network2) == {}
        cls.rm_namespace(network)
        assert cls.namespace_exists(network) == False
        cls.rm_namespace(network2)
        assert cls.namespace_exists(network2) == False
        
        return {'success': True, 'msg': 'Namespace tests passed.'}
    

    @classmethod
    def server_exists(cls, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = cls.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists
    

    def clean(cls, network='local'):
        namespace = cls.get_namespace(network=network)
        address2name = {}
        
        for name, address in address2name.items():
            if address in address2name.values():
                if len(address2name[address]) < len(name):
                    namespace[address] = name
            else:
                address2name[address] = name

        namespace = {v:k for k,v in address2name.items()}
        return namespace
            
    




import commune as c
from typing import *

# THIS IS WHAT THE INTERNET IS, A BUNCH OF NAMESPACES, AND A BUNCH OF SERVERS, AND A BUNCH OF MODULES.
# THIS IS THE INTERNET OF INTERNETS.
class Namespace(c.Module):


    remote_modules_path ='remote_modules'

    # the default
    network : str = 'local'



    @classmethod
    def namespace(cls, search=None,
                   network:str = 'local',
                     update:bool = False, 
                     netuid=None, 
                     max_age:int = None, **kwargs) -> dict:
        
        network = network or 'local'
        if netuid != None:
            network = f'subspace.{netuid}'

        namespace = cls.get(network, {}, max_age=max_age)
        if 'subspace' in network:
            if '.' in network:
                network, netuid = network.split('.')
            else: 
                netuid = netuid or 0
            if c.is_int(netuid):
                netuid = int(netuid)
            namespace = c.module(network)().namespace(search=search, 
                                                 update=update, 
                                                 netuid=netuid,
                                                 **kwargs)
        elif network == 'local':
            if update:
                namespace = cls.build_namespace(network=network)  

   
        namespace = {k:v for k,v in namespace.items() if 'Error' not in k} 
        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in k}
        
        if network == 'local':
            namespace = {k: '0.0.0.0:' + v.split(':')[-1] for k,v in namespace.items() }
        
        namespace = dict(sorted(namespace.items(), key=lambda x: x[0]))

        return namespace
    
    namespace = namespace

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
        address2name = {v: k for k, v in namespace.items()}
        namespace = {v:k for k,v in address2name.items()}
        assert isinstance(namespace, dict), 'Namespace must be a dict.'
        return cls.put(network, namespace)        
    
    add_namespace = put_namespace
    

    @classmethod
    def rm_namespace(cls,network:str) -> None:
        if cls.exists(network):
            cls.rm(network)
            return {'success': True, 'msg': f'Namespace {network} removed.'}
        else:
            return {'success': False, 'msg': f'Namespace {network} not found.'}
    @classmethod
    def name2address(cls, name:str, network:str=network ):
        namespace = cls.namespace(network=network)
        address =  namespace.get(name, None)
        ip = c.ip()
    
        address = address.replace(c.default_ip, ip)
        assert ip in address, f'ip {ip} not in address {address}'
        return address
    
    @classmethod
    def address2name(cls, name:str, network:str=network ):
        namespace = cls.namespace(network=network)
        address2name = {v: k for k, v in namespace.items()}
        return address2name
    
    @classmethod
    def networks(cls) -> dict:
        return [p.split('/')[-1].split('.')[0] for p in cls.ls()]
    
    @classmethod
    def namespace_exists(cls, network:str) -> bool:
        return cls.exists(network)


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
    def build_namespace(cls,
                        timeout:int = 10,
                        network:str = 'local', 
                        verbose=True)-> dict:
        '''
        The module port is where modules can connect with each othe.
        When a module is served "module.serve())"
        it will register itself with the namespace_local dictionary.
        '''
        namespace = {}
        ip = c.ip()
        addresses = [ip+':'+str(p) for p in c.used_ports()]
        future2address = {}
        for address in addresses:
            f = c.submit(c.call, params=[address+'/server_name'], timeout=timeout)
            future2address[f] = address
        futures = list(future2address.keys())
        c.print(f'Updating namespace {network} with {len(futures)} addresses')

        try:
            for f in c.as_completed(futures, timeout=timeout):
                address = future2address[f]
                try:
                    name = f.result()
                    namespace[name] = address
                    c.print(f'Updated {name} to {address}', color='green', verbose=verbose)
                except Exception as e:
                    c.print(f'Error {e} with {address}', color='red', verbose=verbose)
        except Exception as e:
            c.print(f'Timeout error {e}', color='red', verbose=verbose)

        cls.put_namespace(network, namespace)
        
        return namespace

    
    @classmethod
    def migrate_namespace(cls, network:str='local'):
        namespace = cls.get_json('local_namespace', {})
        cls.put_namespace(network, namespace)

    @classmethod
    def merge_namespace(cls, from_network:str, to_network:str, module = None):
        from_namespace = cls.namespace(network=from_network)
        if module == None:
            module = c.module(from_network)

        to_namespace = cls.namespace(network=to_network)
        to_namespace.update(from_namespace)
        cls.put_namespace(to_network, to_namespace)
        return {'success': True, 'msg': f'Namespace {from_network} merged into {to_network}.'}

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
    def add_servers(cls, *servers, network:str='local', **kwargs):
        if len(servers) == 1 and isinstance(servers[0], list):
            servers = servers[0]
        responses = []
        for server in servers:
            try:
                response = cls.add_server(server, network=network)
                responses.append(response)
            except Exception as e:
                e = c.detailed_error(e)
                c.print(f'Could not add {e} to {network} modules. {e}', color='red')
                responses.append({'success': False, 'msg': f'Could not add {server} to {network} modules. {e}'})

        return responses

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
    def server2info(cls, *args, **kwargs):
        return {m['name']:m for m in cls.infos(*args, **kwargs)}
    
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
            remote_modules = cls.get(cls.remote_modules_path, {})
            remote_modules.pop(name, None)
            servers = cls.servers(network=network)
            assert cls.server_exists(name, network=network) == False, f'{name} still exists'
            return {'success': True, 'msg': f'removed {address} to remote modules', 'servers': servers, 'network': network}
        else:
            return {'success': False, 'msg': f'{name} does not exist'}

    
    @classmethod
    def servers(cls, search=None, network:str = 'local', **kwargs):
        namespace = cls.namespace(search=search, network=network, **kwargs)
        return list(namespace.keys())

    @classmethod
    def has_server(cls, name:str, network:str = 'local', **kwargs):
        return cls.server_exists(name, network=network)
    
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
        network = 'test'
        network2  = 'test2'
        cls.rm_namespace(network)
        cls.rm_namespace(network2)

        namespace = cls.namespace(network=network)
        assert cls.namespace(network=network) == {}, f'Namespace not empty., {namespace}'
        cls.register_server('test', 'test', network=network)
        assert cls.namespace(network=network) == {'test': 'test'}, f'Namespace not updated. {cls.namespace(network=network)}'

        assert cls.namespace(network2) == {}
        cls.register_server('test', 'test', network=network2)
        assert cls.namespace(network=network) == {'test': 'test'}, f'Namespace not restored. {cls.namespace(network=network)}'
        cls.deregister_server('test', network=network2)
        assert cls.namespace(network2) == {}
        cls.rm_namespace(network)
        assert cls.namespace_exists(network) == False
        cls.rm_namespace(network2)
        assert cls.namespace_exists(network2) == False
        
        return {'success': True, 'msg': 'Namespace tests passed.'}
    

    @classmethod
    def build_namespace(cls,
                        timeout:int = 2,
                        network:str = 'local', 
                        verbose=True)-> dict:
        '''
        The module port is where modules can connect with each othe.
        When a module is served "module.serve())"
        it will register itself with the namespace_local dictionary.
        '''
        namespace = {}
        ip = c.ip()
        addresses = [ip+':'+str(p) for p in c.used_ports()]
        future2address = {}
        for address in addresses:
            f = c.submit(c.call, params=[address+'/server_name'], timeout=timeout)
            future2address[f] = address
        futures = list(future2address.keys())
        c.print(f'Updating namespace {network} with {len(futures)} addresses')

        try:
            for f in c.as_completed(futures, timeout=timeout):
                address = future2address[f]
                try:
                    name = f.result()
                    if 'Internal Server Error' in name:
                        raise Exception(name)
                    if isinstance(name, dict) and 'error' in name:
                        c.print(f'Error {name} with {address}', color='red', verbose=verbose)
                    else:
                        namespace[name] = address
                    c.print(f'Updated {name} to {address}', color='green', verbose=verbose)
                except Exception as e:
                    c.print(f'Error {e} with {address}', color='red', verbose=verbose)
        except Exception as e:
            c.print(f'Timeout error {e}', color='red', verbose=verbose)

        cls.put_namespace(network, namespace)

            
        return namespace

    
    @classmethod
    def server_exists(cls, name:str, network:str = None,  prefix_match:bool=False, **kwargs) -> bool:
        servers = cls.servers(network=network, **kwargs)
        if prefix_match:
            server_exists =  any([s for s in servers if s.startswith(name)])
            
        else:
            server_exists =  bool(name in servers)

        return server_exists
    

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
            

    @classmethod
    def dashboard(cls):
        return cls.namespace()
    


Namespace.run(__name__)



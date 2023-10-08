import commune as c
from typing import *

class Namespace(c.Module):
    network : str = 'local'

    @classmethod
<<<<<<< HEAD
    def register_server(cls, name:str, address:str, network:str=network) -> None:
=======
    def register_server(cls, name:str, address:str, network=network) -> None:
>>>>>>> 59894203cf3bd502839dbaff9d41ada1ebe04228
        namespace = cls.get_namespace(network)
        namespace[name] = address
        cls.put_namespace(network, namespace)
        return {'status': 'success', 'msg': f'Block {name} registered to {network}.'}

    @classmethod
    def deregister_server(cls, name:str, network=network) -> Dict:

        namespace = cls.get_namespace(network)
        if name in namespace:
            del namespace[name]
            cls.put_namespace(network, namespace)
            return {'status': 'success', 'msg': f'Block {name} deregistered.'}
        else:
            return {'status': 'failure', 'msg': f'Block {name} not found.'}

    @classmethod
<<<<<<< HEAD
    def get_block(cls, name:str, network:str=network) -> dict:
=======
    def get_block(cls, name:str, network=network) -> dict:
>>>>>>> 59894203cf3bd502839dbaff9d41ada1ebe04228
        namespace = cls.get_namespace(network)
        return namespace.get(name, None)

    @classmethod
    def get_namespace(cls, network:str) -> dict:
        return cls.get(network, {})
    @classmethod
    def put_namespace(cls, network:str, namespace:dict) -> None:
        assert isinstance(namespace, dict), 'Namespace must be a dict.'
        cls.put(network, namespace)
        return {'status': 'success', 'msg': f'Namespace {network} updated.'}

    @classmethod
    def rm_namespace(cls,network:str) -> None:
        if cls.exists(network):
            cls.rm(network)
            return {'status': 'success', 'msg': f'Namespace {network} removed.'}
        else:
            return {'status': 'failure', 'msg': f'Namespace {network} not found.'}
    
    @classmethod
<<<<<<< HEAD
    def namespaces(cls) -> dict:
=======
    def networks(cls) -> dict:
>>>>>>> 59894203cf3bd502839dbaff9d41ada1ebe04228
        return [p.split('/')[-1].split('.')[0] for p in cls.ls()]
    
    @classmethod
    def namespace_exists(cls, network:str) -> bool:
        return cls.exists(network)

    @classmethod
    def test(cls):
        network = 'test'
        network2  = 'test2'
        cls.rm_namespace(network)
        cls.rm_namespace(network2)

<<<<<<< HEAD
        assert cls.get_namespace(network) == {}, f'Namespace not empty. {cls.get_namespace(network)}'
=======
        assert cls.get_namespace(network) == {}, 'Namespace not empty.'
>>>>>>> 59894203cf3bd502839dbaff9d41ada1ebe04228
        cls.register_server('test', 'test', network=network)
        assert cls.get_namespace(network) == {'test': 'test'}, f'Namespace not updated. {cls.get_namespace(network)}'

        assert cls.get_namespace(network2) == {}
        cls.register_server('test', 'test', network=network2)
        assert cls.get_namespace(network) == {'test': 'test'}, f'Namespace not restored. {cls.get_namespace(network)}'
        cls.deregister_server('test', network=network2)
        assert cls.get_namespace(network2) == {}
        cls.rm_namespace(network)
        assert cls.namespace_exists(network) == False
        cls.rm_namespace(network2)
        assert cls.namespace_exists(network2) == False
        
        return {'status': 'success', 'msg': 'Namespace tests passed.'}
    
<<<<<<< HEAD
    @classmethod
    def modules(cls, network:List=network) -> List[str]:
        return list(cls.get_namespace(network=network).keys())
    
    @classmethod
    def addresses(cls, network:str=network) -> List[str]:
        return list(cls.get_namespace(network=network).values())
    
    @classmethod
    def module_exists(cls, module:str, network:str=network) -> bool:
        namespace = cls.get_namespace(network=network)
        return bool(module in namespace)
=======
    

    
>>>>>>> 59894203cf3bd502839dbaff9d41ada1ebe04228


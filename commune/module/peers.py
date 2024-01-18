import commune as c
from typing import *

class Peers(c.Module):
    @classmethod
    def add_peer(cls, *args, **kwargs)-> List:
        loop = cls.get_event_loop()
        peer = loop.run_until_complete(cls.async_add_peer(*args, **kwargs))
        return peer
    
    @classmethod
    def add_peers(cls, *peer_addresses, **kwargs): 
        if len(peer_addresses) == 0:
            peer_addresses = cls.boot_peers()
            
        if len(peer_addresses) == 1 and isinstance(peer_addresses[0], list):
            peer_addresses = peer_addresses[0]
        jobs = []
        for peer_address in peer_addresses:
            job = cls.async_add_peer(peer_address, **kwargs)
            jobs += [job]
            
        loop = cls.get_event_loop()
        peers = loop.run_until_complete(asyncio.gather(*jobs))
        peers = [peer for peer in peers if peer != None]
        return {'added_peers': peers, 'msg': f'Added {len(peers)} peers'}

    @classmethod
    def peer_registry(cls, peers=None, update: bool = False):
        if update:
            if peers == None:
                peers = cls.peers()
            cls.add_peers(peers)
        
        peer_registry = c.get('peer_registry', {})
        return peer_registry

    @classmethod
    async def async_add_peer(cls, 
                             peer_address,
                             network = 'local',
                             timeout:int=1,
                             verbose:bool = True,
                             add_peer = True):
        
        peer_registry = await c.async_get_json('peer_registry', default={}, root=True)


        peer_info = await c.async_call(module=peer_address, 
                                              fn='info',
                                              include_namespace=True, 
                                              timeout=timeout)
        
        if add_peer:
            await c.async_call(module=peer_address, 
                                              fn='add_peer',
                                              args=[cls.root_address],
                                              include_namespace=True, 
                                              timeout=timeout)
        

        if 'error' in peer_info:
            if verbose:
                c.print(f'Error adding peer {peer_address} due to {peer_info["error"]}',color='red')
            return None    
        else:
            if verbose:
                c.print(f'Successfully added peer {peer_address}', color='green')
        
            
        assert isinstance(peer_info, dict)
        assert 'address' in peer_info
        assert 'namespace' in peer_info
        
        peer_ip = ':'.join(peer_info['address'].split(':')[:-1])
        peer_port = int(peer_info['address'].split(':')[-1])
        
        # relace default local ip with external_ip
        peer_info['namespace'] = {k:v.replace(c.default_ip,peer_ip) for k,v in peer_info['namespace'].items()}

        peer_registry[peer_address] = peer_info
            
        await c.async_put_json('peer_registry', peer_registry, root=True)
        
        return peer_registry
    
    @classmethod
    def ls_peers(cls, update=False):
        peer_registry = cls.get_json('peer_registry', default={})
        return list(peer_registry.keys())
      
    @classmethod
    def peers(cls, update=False):
        peer_registry = cls.peer_registry(update=update)
        return list(peer_registry.keys())


    @classmethod
    def add_peer(cls, *args, **kwargs)-> List:
        loop = cls.get_event_loop()
        peer = loop.run_until_complete(cls.async_add_peer(*args, **kwargs))
        return peer
    
    @classmethod
    def rm_peers(cls, peer_addresses: list = None):
        rm_peers = []
        if peer_addresses == None:
            peer_addresses = cls.peers()
        if isinstance(peer_addresses, str):
            peer_addresses = [peer_addresses]
        for peer_address in peer_addresses:
            
            rm_peers.append(cls.rm_peer(peer_address))
        return rm_peers
            

    
    @classmethod
    def rm_peer(cls, peer_address: str):
        peer_registry = c.get_json('peer_registry', default={})
        result = peer_registry.pop(peer_address, None) 
        if result != None:
            result = peer_address      
            cls.put_json('peer_registry', peer_registry, root=True)
        return result

    @classmethod
    def boot_peers(cls) -> List[str]: 
        return cls.get('boot_peers', [])
       

    @classmethod
    def get_peer_info(cls, peer: Union[str, 'Module']) -> Dict[str, Any]:
        if isinstance(peer, str):
            peer = cls.connect(peer)
            
        info = peer.info()
        return info
    
        
    @classmethod
    def get_peer_addresses(cls, ip:str = None  ) -> List[str]:
        used_local_ports = cls.get_used_ports() 
        if ip == None:
            ip = c.default_ip
        peer_addresses = []
        for port in used_local_ports:
            peer_addresses.append(f'{ip}:{port}')
            
        return peer_addresses
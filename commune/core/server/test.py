

import commune as c
import os
import torch, time

Server = c.mod('server')
class Test(Server):
    def test_serializer(self):
        return c.mod('serializer')().test()  

    def get_info(self, name:str, timeout:int = 60, interval:int = 1):
        elapsed_seconds = 0
        while elapsed_seconds < timeout:
            try:
                info = c.call(name + '/info')
                if 'key' in info:
                    return info
            except Exception as e:
                print(c.detailed_error(e))
            time.sleep(interval)
            elapsed_seconds += interval
            print(f'waiting for {name} to be available')
        raise TimeoutError(f"Timeout waiting for {name} to be available after {timeout} seconds")

    def test_server(self, 
                    server = 'module::test_serving', 
                    key="test_deployer"):
        c.serve(server, key=key)
        info = self.get_info(server)
        assert info['key'] == c.key(key).ss58_address
        c.kill(server)
        return {'success': True, 'msg': 'server test passed'}

    def test_executor(self):
        return c.mod('executor')().test()

    def test_auth(self, auths=['auth.jwt', 'auth']):
        for auth in auths:
            print(f'testing {auth}')
            c.mod(auth)().test()
        return {'success': True, 'msg': 'server test passed', 'auths': auths}


    def test_role(self, address:str=None, role:str='viber', max_age:int = 60, update:bool = False):
        """
        test the role of the address
        """
        address = address or c.get_key('test').ss58_address
        self.add_role(address, role, max_age=max_age, update=update)
        assert self.get_role(address, max_age=max_age, update=update) == role, f"Failed to add {address} to {role}"
        self.remove_role(address, role, max_age=max_age, update=update)
        assert self.get_role(address, max_age=max_age, update=update) == 'public', f"Failed to remove {address} from {role}"
        return {'roles': True, 'address': address , 'roles': self.get_role(address, max_age=max_age, update=update)}


    def test_blacklist(self,  max_age:int = 60, update:bool = False):  
        """
        check if the address is blacklisted
        """
        key = c.get_key('test').ss58_address
        self.blacklist_user(key, max_age=max_age, update=update)
        assert key in self.blacklist(max_age=max_age, update=update), f"Failed to add {key} to blacklist"
        self.unblacklist_user(key, max_age=max_age, update=update)
        assert key not in self.blacklist(max_age=max_age, update=update), f"Failed to remove {key} from blacklist"
        return {'blacklist': True, 'user': key , 'blacklist': self.blacklist(max_age=max_age, update=update)}

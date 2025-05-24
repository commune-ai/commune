

import commune as c
import os
import torch, time

Server = c.module('server')
class Test(Server):
    def test_serializer(self):
        return c.module('serializer')().test()  
    def test_server(self, name = 'module::test_serving', deployer='module::deployer'):
        module = c.serve(name, key=deployer)
        module = c.connect(name)
        r = module.info()
        r2 = c.call(name+'/info')
        c.print(r, r2)
        assert c.hash(r['key']) == c.hash(r2['key']), f"Failed to get key {r['key']} != {r2['key']}"
        deployer_key = c.get_key(deployer)
        assert r['key'] == deployer_key.ss58_address
        print(r)
        assert 'name' in r, f"get failed {r}"
        c.kill(name)
        assert name not in c.servers(update=True), f"Failed to kill {name}"
        return {'success': True, 'msg': 'server test passed'}
    def test_executor(self):
        return c.module('executor')().test()

    def test_auth(self, auths=['auth.jwt', 'auth']):
        for auth in auths:
            print(f'testing {auth}')
            c.module(auth)().test()
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

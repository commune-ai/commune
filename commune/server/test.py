

import commune as c
import os
import torch, time
class Test:
    def test_serializer(self):
        return c.module('serializer')().test()  
    def test_server(self, name = 'module::test_serving', deployer='module::deployer'):
        module = c.serve(name, key=deployer)
        module = c.connect(name)
        r = module.info()
        r2 = c.call(name+'/info')
        print(r, r2)
        assert c.hash(r) == c.hash(r2)
        deployer_key = c.get_key(deployer)
        assert r['key'] == deployer_key.ss58_address
        print(r)
        assert 'name' in r, f"get failed {r}"
        c.kill(name)
        assert name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}
    def test_executor(self):
        return c.module('executor')().test()

    def test_auth(self, auths=['jwt', 'base']):
        auths = c.get_modules(search='server.auth.')
        for auth in auths:
            c.module(auth)().test()
        return {'success': True, 'msg': 'server test passed', 'auths': auths}




import commune as c
import os
import torch, time

class Test:
    def test_serializer(self):
        self = c.module('serializer')()
        data_list = [
            torch.ones(1000),
            torch.zeros(1000),
            torch.rand(1000), 
            [1,2,3,4,5],
            {'a':1, 'b':2, 'c':3},
            'hello world',
            c.df([{'name': 'joe', 'fam': 1}]),
            1,
            1.0,
            True,
            False,
            None

        ]
        for data in data_list:
            t1 = time.time()
            ser_data = self.serialize(data)
            des_data = self.deserialize(ser_data)
            des_ser_data = self.serialize(des_data)
            t2 = time.time()

            duration = t2 - t1
            emoji = '✅' if str(des_ser_data) == str(ser_data) else '❌'
            print(type(data),emoji)
        return {'msg': 'PASSED test_serialize_deserialize'}


    def test_basics(self) -> dict:
        servers = c.servers()
        c.print(servers)
        name = f'module::test_basics'
        c.serve(name)
        assert name in c.servers()
        c.kill(name)
        assert name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}

    def test_serving(self, name = 'module::test_serving', deployer='module::deployer'):
        module = c.serve(name, key=deployer)
        module = c.connect(name)
        r = module.info()
        r2 = c.call(name+'/info')
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


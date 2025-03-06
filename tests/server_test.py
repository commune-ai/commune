

import commune as c
import os

def test_serializer():
    self = c.module('serializer')()
    import torch, time
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


def test_basics() -> dict:
    servers = c.servers()
    c.print(servers)
    name = f'module::test'
    c.serve(name)
    assert name in c.servers()
    c.kill(name)
    assert name not in c.servers()
    return {'success': True, 'msg': 'server test passed'}

def test_serving(name = 'module::test'):
    module = c.serve(name)
    module = c.connect(name)
    r = module.info()
    assert 'name' in r, f"get failed {r}"
    c.kill(name)
    assert name not in c.servers()
    return {'success': True, 'msg': 'server test passed'}

def test_serving_with_different_key(module = 'module', timeout=2):
    tag = 'test_serving_with_different_key'
    key_name = module + '::'+ tag
    module_name =  module + '::'+ tag + '_b' 
    if not c.key_exists(key_name):
        key = c.add_key(key_name)
    c.print(c.serve(module_name, key=key_name))
    key = c.get_key(key_name)
    c.sleep(2)
    info = c.call(f'{module_name}/info', timeout=timeout)
    assert info.get('key', None) == key.ss58_address , f" {info}"
    c.kill(module_name)
    c.rm_key(key_name)
    assert not c.key_exists(key_name)
    assert not c.server_exists(module_name)
    return {'success': True, 'msg': 'server test passed'}

def test_executor():
    return c.module('executor')().test()


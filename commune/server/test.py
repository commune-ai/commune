

import commune as c

class Test(c.Module):

    @classmethod
    def test_basics(cls) -> dict:
        servers = c.servers()
        c.print(servers)
        tag = 'test'
        module_name = c.serve(module='module', tag=tag)['name']
        c.wait_for_server(module_name)
        assert module_name in c.servers()
        c.kill(module_name)
        assert module_name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}
    

    @classmethod
    def test_serving(cls, server_name = 'module::test'):
        if server_name in c.servers():
            c.kill(server_name)
        module = c.serve(server_name)
        c.wait_for_server(server_name)
        module = c.connect(server_name)
        r = module.info()
        assert 'name' in r, f"get failed {v}"
        c.kill(server_name)
        assert server_name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}

    @classmethod
    def test_serving_with_different_key(cls, module = 'module'):
        tag = 'test_serving_with_different_key'
        key_name = module + '::'+ tag
        module_name =  module + '::'+ tag + '_b' 
        if not c.key_exists(key_name):
            key = c.add_key(key_name)
        if c.server_exists(module_name):
            c.kill(module_name)
        while c.server_exists(module_name):
            c.sleep(1)
        c.print(c.serve(module_name, key=key_name))
        key = c.get_key(key_name)
        while not c.server_exists(module_name):
            c.sleep(1)
            c.print('waiting for server {}'.format(module_name))
        info = c.connect(module_name).info()
        assert info['key'] == key.ss58_address, f"key failed {key.ss58_address} != {info['key']}"
        c.kill(module_name)
        c.rm_key(key_name)
        assert not c.key_exists(key_name)
        assert not c.server_exists(module_name)
        return {'success': True, 'msg': 'server test passed'}


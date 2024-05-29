

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
    def test_serving(cls):
        server_name = 'module::test'
        module = c.serve(server_name)
        c.wait_for_server(server_name)
        
        module = c.connect(server_name)

        module.put("hey",1)
        v = module.get("hey")
        c.print(v)
        assert v == 1, f"get failed {module.get('hey')}"
        
        c.kill(server_name)
        return {'success': True, 'msg': 'server test passed'}


    @classmethod
    def test_serving_with_different_key(cls, module_name = 'storage::test', key_name='storage::test2'):
        c.add_key(key_name)
        module = c.serve(module_name, key=key_name)
        key = c.get_key(key_name)
        c.sleep(1)
        module = c.connect(module_name)
        info = module.info()

        assert info['ss58_address'] == key.ss58_address, f"key failed {key.ss58_address} != {info['ss58_address']}"
        c.kill(module_name)

        c.rm_key(key_name)
        return {'success': True, 'msg': 'server test passed'}




import commune as c


def test_basics() -> dict:
    servers = c.servers()
    c.print(servers)
    name = f'module::test'
    c.serve(name)
    c.kill(name)
    assert name not in c.servers()
    return {'success': True, 'msg': 'server test passed'}

def test_serving(name = 'module::test'):
    module = c.serve(name)
    module = c.connect(name)
    r = module.info()
    assert 'name' in r, f"get failed {r}"
    c.kill(name)
    assert name not in c.servers(update=1)
    return {'success': True, 'msg': 'server test passed'}

def test_serving_with_different_key(module = 'module', timeout=10):
    tag = 'test_serving_with_different_key'
    key_name = module + '::'+ tag
    module_name =  module + '::'+ tag + '_b' 
    if not c.key_exists(key_name):
        key = c.add_key(key_name)
    c.print(c.serve(module_name, key=key_name))
    key = c.get_key(key_name)
    c.sleep(2)
    info = c.call(f'{module_name}/info', timeout=2)
    assert info.get('key', None) == key.ss58_address , f" {info}"
    c.kill(module_name)
    c.rm_key(key_name)
    assert not c.key_exists(key_name)
    assert not c.server_exists(module_name)
    return {'success': True, 'msg': 'server test passed'}
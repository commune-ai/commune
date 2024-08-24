# Description: This is a test file
import commune as c
class Tester(c.Module):
    def __init__(self):
        pass
    def test(self, module='module::test', resolve_server=False):
        if resolve_server:
            server_exists= c.server_exists(module)
            if server_exists:
                c.kill(module)
            while c.server_exists(module):
                pass
            print(f'server_exists: {server_exists}')
            if not server_exists:
                c.serve(module)
            while not c.server_exists(module):
                pass

        client = c.connect(module)
        info = client.info()
        assert info['name'] == module
        assert info['address'] == c.get_address(module)
        assert info['key'] == c.get_key(module).ss58_address
        if c.server_exists(module):
            c.kill(module)
        return info
        
import commune as c

class Server(c.Module):

    @classmethod
    def test(cls) -> dict:
        servers = c.servers()
        c.print(servers)
        tag = 'test'
        module_name = c.serve(module='module', tag=tag)
        c.wait_for_server(module_name)
        assert module_name in c.servers()

        c.kill(module_name)
        assert module_name not in c.servers()
        return {'success': True, 'msg': 'server test passed'}
import commune as c 



class Test(c.Module):

    def test(self):
        servers = c.servers()
        c.print(servers)
        module_name = 'module::test'
        module = c.serve(module='module', name=module_name)
        c.wait_for_server(module)
        assert module_name in c.servers()

        c.kill(module_name)
        assert module_name not in c.servers()
        c.print(c.print(servers))
        


    



import commune as c 



class Test(c.Module):

    def test(self):
        servers = c.servers()
        c.print(servers)
        tag = 'test'
        module_name = c.serve(module='module', tag=tag)
        c.wait_for_server(module_name)
        assert module_name in c.servers()

        c.kill(module_name)
        assert module_name not in c.servers()
        c.print(c.print(servers))
        


    



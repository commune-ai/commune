


import commune as c

class Test:
    core_modules = ['key', 'chain', 'server', 'vali']

    def test_key(self):
        return c.testmod('key')
    def test_chain(self):
        return c.testmod('chain')
    
    def test_server(self):
        return c.testmod('server')
    def test_vali(self):
        return c.testmod('vali')

    @classmethod
    def add_globals(cls, globals):
        fns = [fn for fn in dir(cls) if fn.startswith('test_')]
        for fn in fns:
            globals[fn] = getattr(cls(), fn)
        return True


if __name__ == '__main__':
    Test.add_globals(globals())




import commune as c
for module in c.test_modules:
    def test_module(module=module):
        return c.test_module(module)
    globals()['test_' + module] = test_module
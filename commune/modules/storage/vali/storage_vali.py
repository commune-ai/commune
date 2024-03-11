import commune as c
Vali = c.module('vali')
class StorageVali(Vali):
    def __init__(self, search='storage', network='local', **kwargs):

        config = self.set_config(kwargs=locals())
        self.init_vali(config)
        c.print('init', self.config)

    def score_module(self, module) -> int:
        key = 'test::vali'
        value = 1
        module.put(key, value, refresh=True)
        new_value = module.get(key)
        assert str(value) == str(new_value), f'Error: {value} != {new_value}'
        w = 1
        return w

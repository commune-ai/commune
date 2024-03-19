import commune as c

class StorageVali(c.module('vali')):

    def score_module(self, module, key='test::key',  value = 1) -> int:
        module.put(key, value)
        new_value = module.get(key)
        assert str(value) == str(new_value), f'Error: {value} != {new_value}'
        return 1

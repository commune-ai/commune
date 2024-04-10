import commune as c

class StorageVali(c.module('vali')):

    def score_module(self, module) -> int:
        value = c.random_int()
        key = f'test::vali'
        module.put_item(key, value)
        new_value = module.get_item(key)
        assert value == new_value, f'{value} != {new_value}'
        c.print(w, 'This is the score')
        return w

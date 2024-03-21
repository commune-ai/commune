import commune as c

class StorageVali(c.module('vali')):

    def score_module(self, module, key='test::key',  value = 1) -> int:
        module.put(key, value)
        new_value = module.get(key)
        c.print(module, key)
        if str(value) == str(new_value):
            w = 1
        else:
            w = 0
        c.print(w, 'This is the score')
        return w

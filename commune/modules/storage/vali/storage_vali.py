import commune as c

class StorageVali(c.Module):
    def reward(self, module='storage', **kwargs):
        c.conenct(module, 'reward', kwargs)
        pass
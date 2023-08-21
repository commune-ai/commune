import commune as c

class ValiTextRealfake(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.dataset =  c.module(config.dataset)()


    def run(self):
        print('Base run')

    def score(self, module):
        return self.dataset.score(module)




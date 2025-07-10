class Task:

    def __init__(self, fn='info', params={}):
        self.fn = fn
        self.params = params or {}
        
    def forward(self,mod, fn=None, params=None):
        fn = fn or self.fn
        parmas = params or self.params
        mod = self.mod(mod)
        params = params or {}
        result =  getattr(module, fn)(**params)
        score = 1 if 'url' in result else 0
        return score

    def mod(self, mod_name):
        return c.connect(mod) if isinstance( mod_name, str) else mod_name
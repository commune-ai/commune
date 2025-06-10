class Task:

    def mod(self, mod_name):
        return c.connect(mod) if isinstance( mod_name, str) else mod_name
    def forward(self,mod, fn='info', params=None):
        mod = self.mod(mod)
        params = params or {}
        result =  getattr(module, fn)(**params)
        if 'url' in result:
            score = 1
        else: 
            score = 0
        return {'score': score}
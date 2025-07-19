class Task:

    def __init__(self, fn='info', params={}):
        self.fn = fn
        self.params = params
        
    def forward(self,mod, fn=None, params=None):
        fn = fn or self.fn
        params = params or self.params or {}
        mod = self.mod(mod)
        result =  getattr(mod, fn)(**params)
        return {'fn': fn, 'params': params, 'result': result, 'score': self.score(result)}

    def score(self, x): 
        if 'url' in x and isinstance(x, dict) :
            return 1
        else:
            return 0

    def mod(self, mod):
        return c.connect(mod) if isinstance( mod, str) else mod
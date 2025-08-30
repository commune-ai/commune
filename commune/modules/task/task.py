class Task:
    description = 'Base task for commune'
    features = ['url', 'name', 'score']

    def __init__(self, fn='info', params=None):
        """
        Initialize the task with a function and parameters
        """
        self.fn = fn
        self.params = params or {}

    def forward(self,module, **params):
        params = {**self.params, **params}
        result =  getattr(module, self.fn)(**params)
        if 'url' in result:
            score = 1
        else: 
            score = 0
        return {'score': score, 'result': result, 'params': params}
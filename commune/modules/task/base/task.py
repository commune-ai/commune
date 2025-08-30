class BaseTask:
    description = 'Base task for commune'
    features = ['url', 'name', 'score']

    def __init__(self, fn='info', params=None):
        """
        Initialize the task with a function and parameters
        """
        self.fn = fn
        self.params = params or {}

    def get_params(self,params=None):
        """
        Get a sample from the module
        """
        if params == None:
            params = {
                'fn': self.fn,
                'params': self.params,
            }

        assert isinstance(params, dict)
        assert 'fn' in params
        assert 'params' in params

        return params

    def forward(self,module, params=None):
        params = self.get_params(params)
        result =  getattr(module, params['fn'])(**params['params'])
        assert isinstance(result, dict)
        assert 'name' in result
        return 1
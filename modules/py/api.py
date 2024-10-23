
import commune as c
py = c.module('py')()

def verify(kwargs):
    signature = kwargs['signature']
    assert c.verify(signature), 'Invalid signature.'

class Api(c.Module):
    
    def create_env(self, env, **kwargs):
        verify(kwargs)
        '''Create a virtual environment.'''
        return py.create_env(env)
    
    def remove_env(self, env, **kwargs):
        '''Remove a virtual environment.'''
        ticket = c.verify(**kwargs)
        return py.remove_env(env)
import commune as c

class Executor(c.Module):
    modes = ['thread', 'process']
    
    @classmethod
    def executor(cls, max_workers:int = None, mode:str = 'thread'):
        assert mode in cls.modes, f"mode must be one of {cls.modes}"
        return c.module(f'executor.{mode}')(max_workers=max_workers)
    
    @classmethod
    def test(cls):
        return [c.module('executor.thread').test(), c.module('executor.process').test()]
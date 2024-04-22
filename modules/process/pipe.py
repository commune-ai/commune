import commune as c
import multiprocessing as mp

class Pipe(c.Module):
    """
    
    create a pipe map from and to 
    """
    def __init__(self):
        self.pipes = {}
        self.lock = mp.Lock()

    def add_pipe(self, name, **kwargs):
        with self.lock:
            reader, writer = mp.Pipe(**kwargs)
            self.pipes[name] = {'reader': reader, 'writer': writer, 'kwargs': kwargs}
            return {'success': True, 'name': name, 'msg': 'pipe added'}
    
    def get_pipe(self, name):
        return self.pipes[name]
    
    def get_reader(self, name):
        return self.get_pipe(name)['reader']
    
    def get_writer(self, name):
        return self.get_pipe(name)['writer']

    def rm_pipe(self, name):
        with self.lock:
            pipe = self.pipes.pop(name)
            pipe['reader'].close()
            pipe['writer'].close()
        return {'success': True, 'name': name, 'msg': 'pipe removed'}
        
    def exists(self, name):
        return name in self.pipes
    
    def send(self, name, data):
        with self.lock:
            pipe = self.get_writer(name)
            pipe.send(data)
        return {'success': True, 'name': name, 'msg': 'data sent'}
    def recv(self, name):
        with self.lock:
            pipe = self.get_reader(name)
        return pipe.recv()
    
    @classmethod
    def test(cls):
        pipe = cls()
        pipe.add_pipe('test')
        assert pipe.exists('test'), 'pipe not added'
        pipe.rm_pipe('test')
        assert not pipe.exists('test'), 'pipe not removed'

        # test send and recv
        pipe.add_pipe('test')
        pipe.send('test', 'hello')
        assert pipe.recv('test') == 'hello', 'pipe send and recv not working'
        pipe.rm_pipe('test')
        return {'success': True, 'msg': 'tests for pipe passed', 'module': 'pipe'}

    
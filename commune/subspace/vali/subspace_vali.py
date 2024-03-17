import commune as c

class SubspaceVali(c.Module):
    def __init__(self,network='main' , 
                 tempo=10, max_age=10,):
        self.subspace = c.module('subspace')(network=network)
        self.max_age = max_age
        self.network = network
        self.tempo = tempo

    def score(self,module='subspace', netuid=0) -> int:
        remote_subspace = c.connect(module)
        keys = self.subspace.keys()
        key = c.choice(keys)
        kwargs = {'module_key': key, 'netuid': netuid}
        futures = [c.submit(remote_subspace.get_module, kwargs=kwargs), 
         c.submit(self.subspace.get_module, kwargs=kwargs)]
        remote_module, local_module = c.wait(futures)
        c.print(remote_module, local_module)
        remote_hash = c.hash(remote_module)
        local_hash = c.hash(local_module)
        score = int(remote_hash==local_hash)
        return {'w': score}
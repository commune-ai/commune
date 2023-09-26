import commune


class ModuleF:
    def __init__(self,bro='bro'):
        self.bro = bro  
    def bro(self, fam:str):
        return 'bro'
    
class ModuleG:
    def __init__(self,bro='bro'):
        self.bro = bro  
    def bro(self, fam:str):
        return self.bro
    
    
def test_launch():
    # # print(commune.block.ray_actors())
    commune.launch(ModuleF, mode='ray')
    assert 'ModuleF' in commune.block.ray_actors()
    commune.launch(ModuleG, mode='ray')
    assert 'ModuleG' in commune.block.ray_actors()
    commune.kill_actor('ModuleG')
    assert 'ModuleG' not in commune.block.ray_actors()
    commune.kill_actor('ModuleF')
    
    
    
def test_functions():
    # # print(commune.block.ray_actors())
    module = commune.launch(ModuleF, mode='ray')
    module.bro == 'fam'
    commune.kill_actor('ModuleF')
    
def test_gpu_allocation(gpus:int=1, cpus:int=1):

    commune.launch(ModuleF, mode='ray', gpus=gpus, cpus=cpus)
    print(commune.actors())
    resources = commune.actor_resources('ModuleF')
    assert int(resources['gpus']) == gpus
    assert int(resources['cpus']) == cpus
    commune.kill_actor('ModuleF')
    commune.actor_exists('ModuleF')

if __name__ == '__main__':
    test_gpu_allocation()
    test_launch()
    test_functions()

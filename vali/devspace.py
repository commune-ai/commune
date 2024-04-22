from vali.vali import Vali
from miner.miner import Miner
from subnet.subnet import Subnet
from commune.module.module import Module


class DevMiner(Miner):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

class DevSubnet(Subnet, DevMiner):
    def __init__(self, network = 'local'):
        self.set_config(locals())

        
class Devspace(DevSubnet, Vali, Module):
    def __init__(self, config=None, **kwargs):
        self.init_vali(config=config, kwargs=kwargs)

    def init_vali(self, config=None, module=None, kwargs=None,  **extra_kwargs):
        if module != None:
            assert hasattr(module, 'score'), f'Module must have a config attribute'
            assert callable(module.score), f'Module must have a callable score attribute'
            self.score = module.score
        # initialize the validator
        # merge the config with the default config
        kwargs = kwargs or {}
        kwargs.update(extra_kwargs)
        config = self.set_config(config=config, kwargs=kwargs)
        config = c.dict2munch({**Vali.get_config(), **config})
        c.print(config, 'VALI CONFIG')
        if hasattr(config, 'key'):
            self.key = c.key(config.key)
        self.config = config
        c.thread(self.run_loop)

    init = init_vali
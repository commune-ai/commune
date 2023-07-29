import bittensor
import commune as c

class Dataset(c.Module):
    def __init__(self, config=None, **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)

        bittensor_dataset_config = bittensor.dataset.config()
        config = self.munch({**bittensor_dataset_config.dataset, **config})
        self.print(config)
        self.dataset = bittensor.dataset(config = self.munch(dict(dataset=config)))
        self.config = config
        self.sample()
        

        
        # self.test(self)
        
    # def getattr(self, key):
    #     if hasattr(getattr(self, 'dataset'), key):
    #         return getattr(self.dataset, key)
    #     else:
    #         return getattr(self, key)
    
    # @classmethod
    # def deploy(cls, name=None, tag=None, **kwargs):

    #     return commune.deploy(kwargs=kwargs, name=name, tag=tag)
    
    @classmethod
    def availabe_datasets(cls):
        return cls.get_config().get('available_datasets')
    
    
    @classmethod
    def deploy_fleet(cls, **kwargs):
        tag = kwargs.get('tag', None)
        for d in cls.availabe_datasets():
            c.print(f'Deploying {d}')
            if tag is not None:
                d = f'{d}.{tag}'
            
            cls.deploy(kwargs={'dataset_names': d, **kwargs}, tag=d)
    
    @classmethod
    def check_sample(cls, sample):
        assert isinstance(sample, dict)
        assert 'input_ids' in sample.keys()
        return sample
    
    def sample(self,*args, **kwargs):
        input_ids =  next(self.dataset)
        sample = {'input_ids': input_ids}
        return sample
    
    @classmethod
    def test(cls, *args, module=None, **kwargs):
        cls.print('Testing dataset')
        dataset = cls(*args, **kwargs)
        sample = dataset.sample()
        assert cls.check_sample(sample)
        c.print('Dataset test passed')

if __name__ == "__main__":
    Dataset.run()
        

# Dataset.serve(name='data.bt')
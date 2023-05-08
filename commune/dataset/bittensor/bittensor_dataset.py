import bittensor
import commune

class Dataset(commune.Module):
    def __init__(self, config=None, **kwargs):
        config = self.set_config(config, kwargs=kwargs)

        bittensor_dataset_config = bittensor.dataset.config()
        config = self.munch({**bittensor_dataset_config.dataset, **config})
        self.print(config)
        self.dataset = bittensor.dataset(config = self.munch(dict(dataset=config)))
        self.config = config
        self.sample()
        

        
        self.test(self)
        
    # def getattr(self, key):
    #     if hasattr(getattr(self, 'dataset'), key):
    #         return getattr(self.dataset, key)
    #     else:
    #         return getattr(self, key)
    
    # @classmethod
    # def deploy(cls, name=None, tag=None, **kwargs):

    #     return commune.deploy(kwargs=kwargs, name=name, tag=tag)
    
    
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
    def test(cls, dataset=None, *args, **kwargs):
        if dataset is None:
            dataset = cls(*args, **kwargs)
        cls.print('Testing dataset')
        sample = dataset.sample()
        cls.check_sample(sample)

if __name__ == "__main__":
    Dataset.run()
        

# Dataset.serve(name='data.bt')
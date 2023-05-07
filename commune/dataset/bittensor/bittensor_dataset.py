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
        

        
    # def getattr(self, key):
    #     if hasattr(getattr(self, 'dataset'), key):
    #         return getattr(self.dataset, key)
    #     else:
    #         return getattr(self, key)
    
    # @classmethod
    # def deploy(cls, name=None, tag=None, **kwargs):

    #     return commune.deploy(kwargs=kwargs, name=name, tag=tag)
    
    def sample(self,*args, **kwargs):
        input_ids =  next(self.dataset)
        sample = {'input_ids': input_ids}
        return sample

if __name__ == "__main__":
    Dataset.run()
        

# Dataset.serve(name='data.bt')
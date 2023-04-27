import bittensor
import commune

class Dataset(commune.Module):
    def __init__(self, config=None):
        self.set_config(config)
        self.dataset = bittensor.dataset(**self.config)
        
    def getattr(self, key):
        if hasattr(getattr(self, 'dataset'), key):
            return getattr(self.dataset, key)
        else:
            return getattr(self, key)
            
            
    
    def sample(self,*args, **kwargs):
        input_ids =  next(self.dataset)
        sample = {'input_ids': input_ids}
        return sample

if __name__ == "__main__":
    Dataset.run()
        

# Dataset.serve(name='data.bt')
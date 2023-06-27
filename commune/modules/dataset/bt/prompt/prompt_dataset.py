import commune as c
from datasets import load_dataset


class PromptDataset(c.Module):
    def __init__(self,  config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
        self.dataset = load_dataset(**self.config)
        self.split = self.config.split
        
    def __len__(self):
        return len(self.dataset)
    
        
    def sample(self, idx=None):
        if idx is None:
            idx = c.random_int(len(self))

        return self.dataset[idx]
    
    @classmethod
    def test(cls, *args, module=None, **kwargs):
        cls.print('Testing dataset')
        dataset = cls(*args, **kwargs)
        c.print(dir(dataset))
        sample = dataset.sample()
        print(sample)
        
        assert isinstance(sample, dict)
        return sample
    

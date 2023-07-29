import commune as c
from datasets import load_dataset


class Squad(c.Module):
    def __init__(self, name='squad'):
        self.dataset = load_dataset(name)
        
    def sample(self, idx=4):
        return self.dataset['train'][idx]

    @classmethod
    def test(cls, *args, module=None, **kwargs):
        cls.print('Testing dataset')
        dataset = cls(*args, **kwargs)
        sample = dataset.sample()
        print(sample)
        
        assert isinstance(sample, dict)
        return sample
    
Squad.run(__name__)
import commune as c
import torch

class FinetuneDataset(c.Module,torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        config = self.set_config(kwargs=kwargs)

        kwargs = c.copy(config)
        dataset = kwargs.pop('dataset')
        self.dataset = c.module(dataset)(**kwargs)
        
    def __getitem__(self, idx=None):
        return {'text': c.dict2str(self.dataset.sample(idx=idx))}

    def __len__(self):
        return len(self.dataset)

    def sample(self, *args, **kwargs):
        return 



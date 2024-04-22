import commune as c
import asyncio
import torch
class Dataset(c.Module, torch.utils.data.Dataset):
    mode_shortcuts = {
        'hf': 'text.huggingface',
        'bt': 'text.bittensor',
    }
    def __init__(self,  dataset, config = None, **kwargs):
        config = self.set_config(config, kwargs=)
        self.set_dataset(config)
        self.set_model(config)
        if config.train:
            self.train()
        
    
    @classmethod
    def sample_check(cls, sample):
        return bool(isinstance(sample, dict) and 'input_ids' in sample)
    
    @classmethod
    async def async_sample(cls, dataset = 'dataset.bittensor', max_trials=10, batch_size=1, sequence_length=64, num_batches=10):
        sample = None
        if not hasattr(cls, 'dataset_pool'):
            cls.dataset_pool = c.connect_pool(dataset)

        fail_count = 0
       
        while not cls.sample_check(sample) and fail_count < max_trials:
            if len(cls.dataset_pool) == 0:
                cls.dataset_pool = c.connect_pool(dataset)
            try:
                data_idx =cls.choice(list(range(len(cls.dataset_pool))))
                sample = cls.dataset_pool[data_idx].sample(batch_size=batch_size,
                                        sequence_length=sequence_length)
                
                if not cls.sample_check(sample):
                    raise Exception('Sample check failed')
                sample['input_ids'] = sample['input_ids'][:batch_size, -sequence_length:]
                
                
            except Exception as e:
                fail_count += 1
                del cls.dataset_pool[data_idx]
                cls.print(f'ERROR {e} failed to sample, removing dataset {data_idx}, {len(cls.dataset_pool)} remaining', color='red')
        assert cls.sample_check(sample), f'Failed to sample from {dataset} after {max_trials} trials.'
        return sample
    
    
    @classmethod
    def sample(cls, timeout=2, retries = 3, *args, **kwargs):
        try:
            if timeout:
                # Add timeout to the async_get_sample call
                coro = asyncio.wait_for(cls.async_sample(*args, **kwargs), timeout=timeout)
            else:
                coro = cls.async_sample(*args, **kwargs)
            
            return asyncio.run(coro)
        except asyncio.TimeoutError:
            # Handle the timeout error here
            print("Async function call timed out.")
            if retries > 0:
                return cls.sample(timeout=timeout, retries=retries-1, *args, **kwargs)


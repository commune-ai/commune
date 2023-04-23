import commune

class Dataset(commune.Module):
    mode_shortcuts = {
        'hf': 'text.huggingface',
        'bt': 'text.bittensor',
    }
    def __init__(self, *args,mode='hf', **kwargs):
        mode = self.mode_shortcuts.get(mode, mode)
        module = commune.get_module(f'dataset.{mode}')(*args, **kwargs)
        self.merge(self,module)
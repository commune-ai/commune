

import commune
from typing import List

class Pile(commune.Module):
    num_shards = 29
    default_shards = list(range(num_shards))
    url = 'https://the-eye.eu/public/AI/pile'
    
    def __init__(self,config=None):
        config = self.set_config(config)
        self.set_shards(config.shards)
    
    
    def set_shards(self, shards):
        self.shards_urls = self.get_shard_urls(shards)
            
    @classmethod
    def resolve_shards(cls, shards):
        if isinstance(shards, int):
            shards = list(range(shards))
        assert isinstance(shards, list)
        
        for s in shards:
            assert isinstance(s, int)
            
        return shards

    @classmethod
    def get_shard_urls(cls, shards: List[int] = 29, split='train'):
        shards = cls.resolve_shards(shards)
        shard_urls = []
        for s in shards:
            shard_urls.append(cls.get_shard_url(s, split=split))
        print(shard_urls)
        return shard_urls
        
    
    @classmethod
    def get_shard_url(cls, shard=0, split='train'):
        assert isinstance(shard, int)
        filename =f'{shard}.jsonl.zst' if  bool(shard >= 10) else f'0{shard}.jsonl.zst'
        shard_url = f'{cls.url}/{split}/{filename}'
        return shard_url
    
    @classmethod
    def ls_shards(cls):
        return [p for p in cls.glob('shards') if p.endswith('.jsonl')]
    
    @classmethod
    def get_shard_path(cls, shard:int, split:str='train', ext='jsonl'):
        filename = f'{shard}' if shard >= 10 else f'0{shard}'
        path= cls.resolve_path(f'shards/{filename}.{ext}')
        return path

    resolve_shard_path = get_shard_path
    @classmethod
    def shard_exists(cls, shard:int, split='train')-> bool:
        shard_path = cls.resolve_shard_path(shard,split)
        return bool(shard_path in cls.ls_shards())
    
    
    @classmethod
    def download_shard(cls, 
                       shard:int = 1,
                       split='train', 
                       refresh: bool = False,
                       *args, **kwargs):
        shard_url = cls.get_shard_url( shard=shard, split=split)
        shard_exists = cls.shard_exists(shard=shard, split=split)
        path = cls.resolve_shard_path(shard, split)
            
        if shard_exists and not refresh :
            cls.print(f'THE PILE: shard {shard} for split {split} exists', color='yellow')
            return None
        
        return cls.cmd(f'wget -P {path} {shard_url}', verbose=True, *args, **kwargs)
        
    @classmethod
    def download_fleet(cls, shards=3, split='train'):
        for s in range(shards):
            name = f'task.pile.download.s{s}.{split}'
            cls.deploy(fn='deploy_shard', name=name )
            
        
        
    def get_text(self, shard=1, split='train', path=None ):
        path = self.get_shard_path(shard=shard, split=split) if path == None else path
        with open(path, 'r') as f:
            # Loop through each line in the file
            cnt = 0
            t = commune.timer()
            for line in f:
                # Load the JSON data from the line
                data = json.loads(line)
                # Do something with the data, for example print a specific key
                # print(data['text'])
                yield data[text]
                # break
                

    
    def sample(self):
        return self.get_sample()
    
    
    @classmethod
    def test(cls, *args, **kwargs):
        self = cls(*args,**kwargs)
        print(next(self.get_text()))


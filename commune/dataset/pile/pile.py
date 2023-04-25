

import commune

class Pile(commune.Module):
    num_shards = 29
    default_shards = list(range(num_shards))
    def get_pile(self, shards=None, *args, **kwargs):
        if shards == None:
            shards = self.default_shards
        
        for shard in shards:
            self.get_shard(shard, *args, **kwargs)
        return 
    url = 'https://the-eye.eu/public/AI/pile'
    
    
    @classmethod
    def get_shard_url(cls, shard=0, split='train'):
        assert isinstance(shard, int)
        greater_than_10 = bool(shard > 10)
        filename =f'{shard}.jsonl.zst' if shard>10 else f'0{shard}.jsonl.zst'
        shard_url = f'{cls.url}/{split}/{filename}'
        return shard_url
    
    @classmethod
    def get_shard(cls, shard:int, split='train', *args, **kwargs):
        shard_url = cls.get_shard_url( shard=shard, split=split)
        path = cls.resolve_path(f'pile/shard-{shard}')
        cls.cmd(f'wget -P {path} {shard_url}',  *args, **kwargs)
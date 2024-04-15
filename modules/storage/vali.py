import commune as c

class StorageVali(c.module('vali')):


    whitelist = ['put_item', 'get_item', 'score', 'leaderboard', 'eval']
    
    def __init__(self, netuid= 0,
                    search= 'storage',
                    storage_timeout= 5,
                    tag= 'base',
                    network= 'local',
                    max_workers= 1, **kwargs
    ):
        self.init_vali(locals())

    def score(self, module) -> int:
        value = c.random_int()
        key = f'test_storage/{self.key.ss58_address}'
        module.put_item(key, value)
        new_value = module.get_item(key)
        assert value == new_value, f'{value} != {new_value}'
        return 1
    
    def put_item(self, key: str, value: int) -> int:
        return self.put('storage/' + key, value)
    
    def get_item(self, key: str) -> int:
        return self.get('storage/' + key)
    
    @classmethod
    def testnet(cls) -> int:
        c.serve('storage')
        c.serve('storage.vali', remote=0, debug=1)
        return 1





import commune
from web3.main import Web3

class Hash(commune.Module):
    hash_fn_dict = {
        'keccak': Web3.keccak
    }
    @staticmethod
    def resolve_hash_function(cls, hash_type: str='keccak'):
        hash_fn = cls.hash_fn_dict.get(hash_type)
        assert hash_fn != None, f'hash_fn: {hash_type} is not found'
        return hash_fn


    def hash(cls, x, hash_type: str='keccak',return_string: bool =True,*args,**kwargs):
        x = cls.python2str(x)
        if hash_type == 'keccak':
            hash_output = Web3.keccak(text=x, *args, **kwargs)
            if return_string:
                hash_output = Web3.toHex(hash_output)
            return hash_output
        else:
            raise NotImplemented(hash_type)

    def __call__(self, *args, **kwargs):
        return self.hash(*args, **kwargs)

if __name__ == "__main__":
    Hash.run()
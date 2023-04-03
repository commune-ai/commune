

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


    def hash(cls, data, mode: str='keccak',return_string: bool =True,*args,**kwargs):
        data = cls.python2str(data)
        if mode == 'keccak':
            hash_output = Web3.keccak(text=data, *args, **kwargs)
            if return_string:
                hash_output = Web3.toHex(hash_output)
        elif mode == 'ss58':
            # only works for 32 byte hex strings
            hash_fn = commune.import_object('scalecodec.utils.ss58.ss58_encode')
            ss58_format=42
            try:
                hash_output = hash_fn(data, ss58_format=ss58_format) 
            except ValueError as e:
                data = cls.hash(data, mode='keccak', return_string=return_string, *args, **kwargs)
                hash_output = hash_fn(data, ss58_format=ss58_format)
        else:
            raise NotImplemented(mode)


        return hash_output

    def __call__(self, *args, **kwargs):
        return self.hash(*args, **kwargs)

if __name__ == "__main__":
    Hash.run()
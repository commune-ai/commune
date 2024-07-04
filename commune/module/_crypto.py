import hashlib

class Crypto:
    @classmethod
    def hash(cls, x, mode: str='sha256',*args,**kwargs):
        x = cls.python2str(x)
        if mode == 'keccak':
            return cls.import_object('web3.main.Web3').keccak(text=x, *args, **kwargs).hex()
        elif mode == 'ss58':
            return cls.import_object('scalecodec.utils.ss58.ss58_encode')(x, *args,**kwargs) 
        elif mode == 'python':
            return hash(x)
        elif mode == 'md5':
            return hashlib.md5(x.encode()).hexdigest()
        elif mode == 'sha256':
            return hashlib.sha256(x.encode()).hexdigest()
        elif mode == 'sha512':
            return hashlib.sha512(x.encode()).hexdigest()
        elif mode =='sha3_512':
            return hashlib.sha3_512(x.encode()).hexdigest()
        else:
            raise ValueError(f'unknown mode {mode}')

        #TODO: add quantum resistant hash functions


        return hash_output
    
    @classmethod
    def hash_modes(cls):
        return ['keccak', 'ss58', 'python', 'md5', 'sha256', 'sha512', 'sha3_512']
    



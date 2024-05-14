import commune as c
import hashlib

class Hash(c.Module):
    @classmethod
    def hash(cls, x, mode: str='sha256',*args,**kwargs):
        x = cls.python2str(x)
        if mode == 'keccak':
            return c.import_object('web3.main.Web3').keccak(text=x, *args, **kwargs).hex()
        elif mode == 'ss58':
            return c.import_object('scalecodec.utils.ss58.ss58_encode')(x, *args,**kwargs) 
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

    @classmethod
    def test(cls, x='bro'):
        for mode in cls.hash_modes():
            try:
                cls.print(f'SUCCESS {mode}: x -> {cls.hash(x, mode=mode)}', color='green')
            except Exception as e:
                cls.print(f'FAILED {mode}: x -> {e}', color='red')

    def __call__(self, *args, **kwargs):
        return self.hash(*args, **kwargs)

if __name__ == "__main__":
    Hash.run()
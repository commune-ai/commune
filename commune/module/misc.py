
from typing import List, Union
class Misc:

    @staticmethod
    def chunk(sequence:list = [0,2,3,4,5,6,6,7],
            chunk_size:int=4,
            num_chunks:int= None):
        assert chunk_size != None or num_chunks != None, 'must specify chunk_size or num_chunks'
        if chunk_size == None:
            chunk_size = len(sequence) / num_chunks
        if chunk_size > len(sequence):
            return [sequence]
        if num_chunks == None:
            num_chunks = int(len(sequence) / chunk_size)
        if num_chunks == 0:
            num_chunks = 1
        chunks = [[] for i in range(num_chunks)]
        for i, element in enumerate(sequence):
            idx = i % num_chunks
            chunks[idx].append(element)
        return chunks
    
    @classmethod
    def batch(cls, x: list, batch_size:int=8): 
        return cls.chunk(x, chunk_size=batch_size)

    def cancel(self, futures):
        for f in futures:
            f.cancel()
        return {'success': True, 'msg': 'cancelled futures'}
       
    
    @classmethod
    def cachefn(cls, func, max_age=60, update=False, cache=True, cache_folder='cachefn'):
        import functools
        path_name = cache_folder+'/'+func.__name__
        def wrapper(*args, **kwargs):
            fn_name = func.__name__
            cache_params = {'max_age': max_age, 'cache': cache}
            for k, v in cache_params.items():
                cache_params[k] = kwargs.pop(k, v)

            
            if not update:
                result = cls.get(fn_name, **cache_params)
                if result != None:
                    return result

            result = func(*args, **kwargs)
            
            if cache:
                cls.put(fn_name, result, cache=cache)
            return result
        return wrapper


    @staticmethod
    def round(x:Union[float, int], sig: int=6, small_value: float=1.0e-9):
        import math
        """
        Rounds x to the number of {sig} digits
        :param x:
        :param sig: signifant digit
        :param small_value: smallest possible value
        :return:
        """
        x = float(x)
        return round(x, sig - int(math.floor(math.log10(max(abs(x), abs(small_value))))) - 1)
    
    @classmethod
    def round_decimals(cls, x:Union[float, int], decimals: int=6, small_value: float=1.0e-9):
        import math
        """
        Rounds x to the number of {sig} digits
        :param x:
        :param sig: signifant digit
        :param small_value: smallest possible value
        :return:
        """
        x = float(x)
        return round(x, decimals)
    
    


    @staticmethod
    def num_words( text):
        return len(text.split(' '))
    
    @classmethod
    def random_word(cls, *args, n=1, seperator='_', **kwargs):
        random_words = c.module('key').generate_mnemonic(*args, **kwargs).split(' ')[0]
        random_words = random_words.split(' ')[:n]
        if n == 1:
            return random_words[0]
        else:
            return seperator.join(random_words.split(' ')[:n])

    @classmethod
    def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
        return [text for text in text_list if filter_text in text]



    @staticmethod
    def tqdm(*args, **kwargs):
        from tqdm import tqdm
        return tqdm(*args, **kwargs)

    progress = tqdm



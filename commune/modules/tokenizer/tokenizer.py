
import commune as c
import torch
from typing import Dict, Any, List, Tuple


class Tokenizer(c.Module):
    def __init__(self, tokenizer='gpt2', cache = True,  **kwargs):
        self.tokenizer = self.get_tokenizer(tokenizer=tokenizer, cache=cache, **kwargs)
        self.device = None
        self.tokenizer_cache = {}

    def set_tokenizer(self, tokenizer='gpt2', cache = True,  **kwargs):
        self.tokenizer = self.get_tokenizer(tokenizer=tokenizer, cache=cache, **kwargs)

    tokenizer_cache = {}
    @classmethod
    def get_tokenizer(cl, tokenizer='gpt2', cache = True,  **kwargs):
        if cache and tokenizer in self.tokenizer_cache:
            return cls.tokenizer_cache[tokenizer]
        from transformers import AutoTokenizer
        tokenizer_obj =  AutoTokenizer.from_pretrained(tokenizer,**kwargs)
        if cache:
            cls.tokenizer_cache[tokenizer] = tokenizer_obj
        
    def tokenize(self, text:str,  **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    def detokenize(self, tokens:list,  **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)

    def detokenize(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        text = self.tokenizer.batch_decode(input_ids,**kwargs)  # assume tokenizer.padding_side = 'left'
        return text
    
    @classmethod
    def test(cls,**kwargs):
        text = 'Whadup'
        self  = cls(**kwargs)
        print(self.tokenize(text))
        print(self.detokenize(self.tokenize(text)['input_ids']))

    @classmethod
    def deploy(cls, tokenizer):
        return cls.deploy(tag=tokenizer, kwargs=dict(tokenizer=tokenizer)   )
    
if __name__ == "__main__":
    Tokenizer.run()
    
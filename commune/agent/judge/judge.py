import commune
st = commune.get_streamlit()

from typing import *
class Judge(commune.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_tokenizer(self.config['tokenizer'])


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        tokenizer = tokenizer if tokenizer else self.model_path
        from transformers import AutoTokenizer
        
        if isinstance(tokenizer, str):
            self.config['tokenizer'] = tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer
        
        
        return self.tokenizer



if __name__ == "__main__":
    Judge.run()






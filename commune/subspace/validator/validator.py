import commune
import streamlit as st


st.write(commune.servers())


class Validator(commune.Module):
    
    def __init__(self, 
                 dataset:str = 'dataset',
                 tokenizer: str = 'gpt2',
                 network: str = 'local'):
        self.set_dataset(dataset)
        self.set_tokenizer(tokenizer)
        
    def set_dataset(self, dataset: str) -> None:
        self.dataset = commune.connect(dataset)
        
    def validate(self, value):
        return value
    


    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6B',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
         'gpt3b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b'

         }

        tokenizer = tokenizer if tokenizer else self.model_path
        from transformers import AutoTokenizer
        
        if isinstance(tokenizer, str):
            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
            self.config['tokenizer'] = tokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        self.tokenizer = tokenizer
        

        return self.tokenizer

    
    
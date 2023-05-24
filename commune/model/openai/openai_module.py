
import openai
import os
import torch
from typing import Union, List, Any, Dict
import commune as c
import json
# class OpenAILLM(c.Module):
default_prompt = """
Predict the topk percent for the next token 
params:(tokenizer={tokenizer}, k={k}, text: {text}) 
Output a dict of (token:str, score:int) and do it for 100 tokens.
"""

class OpenAILLM(c.Module):
    def __init__(self,
                 model: str = "text-davinci-003",
                temperature: float=0.9,
                max_tokens: int=10,
                top_p: float=1.0,
                frequency_penalty: float=0.0,
                presence_penalty: float=0.0,
                tokenizer: str = None,
                prompt: str = None,
                api: str = 'OPENAI_API_KEY'
                ):
        self.set_api(api)
        self.set_prompt(prompt)
        self.set_tokenizer(tokenizer)
     
    
    def set_api(self, api: str = None) -> None:
        openai.api_key = os.getenv(api, None)
        
        if isinstance(api, str) and openai.api_key is None:
            openai.api_key = api
        assert openai.api_key is not None, "OpenAI API key not found."

    def forward(self,prompt:str=None,
                params: dict = None,
                return_text: bool = False,
                verbose: bool = True,
                **kwargs) -> str:
        
        
        if 'input_ids' in kwargs: 
            kwargs['text'] = self.decode_tokens(kwargs.pop('input_ids'))
        prompt  = prompt if prompt != None else self.prompt
        prompt = prompt.format(**kwargs)
        params = params if params != None else self.params
        if verbose:
            c.print(f'Running OpenAI LLM with params:', params, color='purple')
            c.print(f" PROMPT: {prompt}", color='yellow')
        
        response = openai.Completion.create(
            prompt=prompt, 
            **params
        )
        output_text = response['choices'][0]['text']
        
        if verbose:
            c.print('Result: ', output_text, 'green')
        if return_text:
            return output_text

        return {'text': output_text}
    
    
    def set_prompt(self, prompt: str):
        
        if prompt == None:
            prompt = self.prompt
        self.prompt = prompt
        assert isinstance(self.prompt, str), "Prompt must be a string"
        self.prompt_variables = self.get_prompt_variables(self.prompt)
    
    prompt = """
        Predict the topk percent for the next token 
        params:(tokenizer={tokenizer}, k={k}, text: {text}) 
        Output a dict of (token:str, score:int) and do it for 100 tokens.
        """
        
    @staticmethod   
    def get_prompt_variables(prompt):
        variables = []
        tokens = prompt.split('{')
        for token in tokens:
            if '}' in token:
                variables.append(token.split('}')[0])
        return variables
    
        

    @classmethod
    def test(cls, **params:dict):
        
        
        
        model = cls(**params)
        cls.print(model.__dict__)
         

    def set_tokenizer(self, tokenizer: str):

        if tokenizer == None and hasattr(self, 'tokenizer'):
            return self.tokenizer
             
        if tokenizer == None:
            tokenizer = 'gpt2'
        from transformers import AutoTokenizer

        if isinstance(tokenizer, str):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        
        tokenizer.pad_token = tokenizer.eos_token 
            
        self.tokenizer = tokenizer
        
        
        return self.tokenizer
    
    
    def decode_tokens(self,input_ids: Union[torch.Tensor, List[int]], **kwargs) -> Union[str, List[str], torch.Tensor]:
        return self.tokenizer.decode(input_ids, **kwargs)
    def encode_tokens(self, 
                 text: Union[str, List[str], torch.Tensor], 
                 return_tensors='pt', 
                 padding=True, 
                 truncation=True, 
                 max_length=256,
                 **kwargs):
        
        return self.tokenizer(text, 
                         return_tensors=return_tensors, 
                         padding=padding, 
                         truncation=truncation, 
                         max_length=max_length)
    @classmethod
    def example(cls):
        model = cls()
        input_ids = model.encode_tokens('Hellow how is it going')['input_ids']
        print(model.forward(input_ids=input_ids[0], tokenizer='gpt2', k=10))
            


        return cls.test(cls.example_params())





if __name__ == '__main__':
    OpenAILLM.run()



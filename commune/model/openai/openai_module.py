
import openai
import os
import torch
from typing import Union, List, Any, Dict
import commune
# class OpenAILLM(commune.Module):
prompt = """
Predict the topk percent for the next token 
params:(tokenizer={tokenizer}, k={k}, text: {text}) 
Output a dict of (token:str, score:int) and do it for 100 tokens.
"""

class OpenAILLM(commune.Module):
    def __init__(self,
                 model: str = "text-davinci-003",
                temperature: float=0.9,
                max_tokens: int=10,
                top_p: float=1.0,
                frequency_penalty: float=0.0,
                presence_penalty: float=0.0,
                tokenizer: str = None,
                prompt: str = None,
                key: str = 'OPENAI_API_KEY'
                ):
        self.set_key(key)
        self.set_params(locals())
        
        
    def set_params(self, params: dict):
        assert isinstance(params, dict), "params must be a dict."
        param_keys = ['model', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']
        
        self.set_prompt(params.pop('prompt'))
        self.set_tokenizer(params.pop('tokenizer'))
        self.params = {}
        for key in param_keys:
            self.params[key] = params[key]
        
        
            
        
            
        
        
        
        
    def set_key(self, key: str = None) -> None:
        openai.api_key = os.getenv(key, None)
        
        if isinstance(key, str) and openai.api_key is None:
            openai.api_key = key
        assert openai.api_key is not None, "OpenAI API key not found."
            
        
    # @classmethod
    # def install_env(cls):
    #     cls.cmd("pip install openai")





    def forward(self,prompt=None,
            params: dict = None,
            return_dict: bool = False,
            return_text_only: bool = True,
            **kwargs
    ) -> str:
        if 'input_ids' in kwargs: 
            kwargs['text'] = self.decode_tokens(kwargs.pop('input_ids'))
        prompt  = prompt if prompt != None else self.prompt
        
        print(prompt)

        prompt = prompt.format(**kwargs)
        
        params = params if params != None else self.params

        commune.print(f"Running OpenAI LLM with params: {params}", 'purple')
        commune.print(f" PROMPT: {prompt}", 'yellow')
        response = openai.Completion.create(
            prompt=prompt, 
            **params
        )
        output_text = response['choices'][0]['text']
        commune.print('Result: '+output_text, 'green')
        
        if return_text_only:
            return output_text
        
        
        if return_dict: 
            return json.loads(text.replace("'", '"'))

        return response
    
    @property
    def prompt(self):
        if hasattr(self, '_prompt'):
            return self._prompt
        
        prompt = """
        
        Hello
        
        """
        
        return prompt
    
    
    @prompt.setter
    def prompt(self, prompt: str):
        self._prompt = prompt
        
    def set_prompt(self, prompt: str):
        self.prompt = prompt
    

    @classmethod
    def test(cls, params:dict):
        
        
        
        model = OpenAILLM(**params)
         

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



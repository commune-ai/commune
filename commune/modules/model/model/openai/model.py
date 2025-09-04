import openai
from typing import *
import commune as c

class OpenAILLM:

    libs = ['openai', 'transformers']
    expose = ['generate']
    
    def __init__(self, api_key = None,
                max_tokens : int= 250,
                max_stats: int= 175,
                tokenizer: str= 'gpt2',
                save:bool = False,
                max_input_tokens: int = 10_000_000,
                max_output_tokens: int = 10_000_000,
                **kwargs
                ):
    
        self.set_config(locals())
        self.birth_time = c.time()
        self.set_api_key(api_key)
        # self.test()
        
    def forward(self,
                prompt:str = 'sup?',
                model:str = 'gpt-4-vision-preview',
                presence_penalty:float = 0.0, 
                frequency_penalty:float = 0.0,
                temperature:float = 0.9, 
                max_tokens:int = 4096, 
                top_p:float = 1,
                choice_idx:int = 0,
                api_key:str = None,
                role:str = 'user',
                history: list = None,
                stream =  False,
                **kwargs) -> str:
        
        t = c.time()
    
        openai.api_key = api_key or self.api_key

        params = dict(
                    model = model,
                    presence_penalty = presence_penalty, 
                    frequency_penalty = frequency_penalty,
                    temperature = temperature, 
                    max_tokens = max_tokens, 
                    top_p = top_p
                    )
        
        messages = [{"role": role, "content": prompt}]
        
        if history:
            messages = history + messages

        response = openai.chat.completions.create(messages=messages, stream=stream, **params)
        
        if stream:
            def stream_response(response):
                for r in response:
                    token = r.choices[choice_idx].delta.content
                    yield token
            return stream_response(response)
        else:
            return response.choices[choice_idx].message.content

    generate = call = forward

    def test(self, mod = None, x:str = 'What is the meaning of life?',**kwargs):
        mod = mod or self
        assert isinstance(mod.generate(x), str)
        return {'success': True, 'msg': 'test'}
    
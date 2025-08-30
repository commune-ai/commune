import openai
from typing import *
import commune as c

class OpenAILLM(c.Module):

    libs = ['openai', 'transformers']
    
    prompt = """{x}"""

    whitelist = ['generate']
    
    def __init__(self, api_key = None,
                max_tokens : int= 250,
                max_stats: int= 175,
                tokenizer: str= 'gpt2',
                save:bool = False,
                prompt:bool = None,
                max_input_tokens: int = 10_000_000,
                max_output_tokens: int = 10_000_000,
                **kwargs
                ):
        

        self.set_config(locals())
        self.birth_time = c.time()
        self.set_api_key(api_key)
        self.set_prompt(prompt)
        # self.test()
        
    @property
    def age(self):
        return c.time() - self.birth_time
        

    def resolve_prompt(self, *args, prompt = None, **kwargs):
        if prompt == None:
            prompt = self.prompt
            prompt_variables  = self.prompt_variables
        else:
            assert isinstance(prompt, str)
            prompt_variables = self.get_prompt_variables(prompt)
        
                    
        if len(args) > 0 :
            assert len(args) == len(prompt_variables), f"Number of arguments must match number of prompt variables: {self.prompt_variables}"
            kwargs = dict(zip(prompt_variables, args))

        for var in prompt_variables:
            assert var in kwargs

        prompt = prompt.format(**kwargs)
        return prompt
    
    
    def generate(self,
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
            output_text = response = response.choices[choice_idx].message.content
            return output_text
    
    
    _stats = None
    _stats_update_time = 0


    forward = call = generate

    @classmethod
    def chat(cls, *args, **kwargs):
        return cls().forward(*args, **kwargs)
        
    @property
    def history(self):
        return self.config.get('history', [])
    @history.setter
    def history(self, history):
        self.config['history'] = history

    def set_prompt(self, prompt: str):
        if prompt == None:
            prompt = self.prompt
        self.prompt = prompt
        assert isinstance(self.prompt, str), "Prompt must be a string"
        self.prompt_variables = self.get_prompt_variables(self.prompt)

        
    @staticmethod   
    def get_prompt_variables(prompt):
        variables = []
        tokens = prompt.split('{')
        for token in tokens:
            if '}' in token:
                variables.append(token.split('}')[0])
        return variables


    @classmethod
    def test(cls, input:str = 'What is the meaning of life?',**kwargs):
        module = cls()
        output = module.generate(input)
        assert isinstance(output, str)
        return {'success': True, 'msg': 'test'}
    
    @classmethod
    def validate(cls, text = 'What is the meaning of life?', max_tokens=10):
        prefix = cls.module_path()
        jobs = []
        servers = c.servers(prefix)
        for s in servers:
            job = c.call(module=s, 
                         fn='forward', 
                         text=text, 
                         temperature=0.0,
                         max_tokens=max_tokens,
                        return_future=True
                        )
            jobs.append(job)
        assert len(jobs) > 0, f'No servers found with prefix {prefix}'
        results = c.gather(jobs)
        response = {}
        for s, result in zip(c.servers(prefix), results):
            response[s] = result

        return response
   
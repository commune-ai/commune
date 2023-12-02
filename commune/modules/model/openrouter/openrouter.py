import commune as c
import requests
import json
import os
from typing import List

class OpenRouterModule(c.Module):
    whitelist = ['generate', 'models']

    def __init__(self,
                url:str = "https://openrouter.ai/api/v1/chat/completions",
                model: str = "openai/gpt-3.5-turbo-16k",
                role: str = "user",
                http_referer: str = "http://localhost:3000",
                api_key: str = 'OPEN_ROUTER_API_KEY',
                x_title: str = "Communne",
                max_history: int = 100,
                **kwargs
                ):
        self.url = url
        self.set_api_key(api_key)
        self.set_models(model)

    

        self.model = model
        self.role = role
        self.http_referer = http_referer
        self.x_title = x_title
        self.max_history = max_history

    def set_models(self, model:str):
        if model == None:
            model = c.choice(self.models())
            self.models = self.self.models()
        elif 'all' in model:
            self.models = self.models()
        elif isinstance(model, str):
            self.models = self.models(search=model)
        elif isinstance(model, list):
            for m in model:
                assert isinstance(m, str), "Model must be a string"
                self.models = self.models(search=m)

        assert isinstance(self.models, list), "Model must be a list"

        return {"status": "success", "model": model, "models": self.models}
        


    def generate(self, content: str, text_only:bool = True, model=None, history=None, trials=1, api_key=None ):


        # trials 
        while trials > 1:
            try:
                response = self.generate(content=content, text_only=text_only, history=history, trials=1)
            except Exception as e:
                e = c.detailed_error(e)
                trials -= 1
                c.print('{t} trials Left')
                c.print(e)
                continue
            
            return response

        assert trials > 0

            
                

        model = model or c.choice(self.models)['id']
        history = history or []

        c.print(f"Generating response with {model}...", color='yellow')

        data = {
                "model": model, 
                "messages": history + [{"role": self.role, "content": content} ]
            }
        

        t1 = c.time()
        response = requests.post(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.http_referer, 
                "X-Title": self.x_title, 
            },

            data=json.dumps(data)
            )
        t2 = c.time()
        latency = t2 - t1
        response = json.loads(response.text)

        c.print(response)

        tokens_per_word = 2
        if 'choices' not in response:
            return response
        output_text = response["choices"][0]["message"]["content"]
        output_tokens = output_text * tokens_per_word

    
        path = f'state/{model}'
        state = self.get(path, {})

        state = {
            'latency': latency,
            'output_tokens': state.get('output_tokens', 0) + self.num_tokens(output_tokens),
            'timestamp': t2,
            'count': state.get('count', 0) + 1,
            'data': [state.get('data', []) + [data]][:self.max_history],
        }

        self.put(path, state)

        if text_only:
            return output_text
            

        return response
    
    prompt = generate

    def set_api_key(self, api_key:str):
        api_key = os.getenv(api_key, None)
        if api_key == None:
            api_keys = self.api_keys()
            assert len(api_keys) > 0, "No API keys found. Please set an API key with OpenRouterModule.set_api_key()"
            api_key = c.choice(api_keys)
        assert isinstance(api_key, str), "API key must be a string"
        self.api_key = api_key   

    def test(self, text = 'Hello', model=None):
        t1 = c.time()
        if model == None:
            model = c.choice(self.models)['id']
        response = self.prompt(text, model=model, text_only=True)
        tokens_per_second = self.num_tokens(response) / (c.time() - t1)
        latency = c.time() - t1
        assert isinstance(response, str)
        return {"status": "success", "response": response, 'latency': latency, 'model': model , 'tokens_per_second': tokens_per_second}
    

    def test_models(self, search=None, timeout=10, models=None):
        models = models or self.models(search=search)
        futures = [c.submit(self.test, kwargs=dict(model=m['id']), timeout=timeout) for m in models]
        results = c.wait(self.test)
        return results
    
    @classmethod
    def model2info(cls, search:str = None):
        models = cls.models(search=search)
        if search != None:
            models =  [m for m in models if search in m['id']]
        return {m['id']:m for m in models}
    
    
    @classmethod
    def models(cls, search:str = None, update=False, path='model'):
        if not update:
            models =  cls.get(path, [])
        if len(models) == 0:

            c.print('Updating models...', color='yellow')
            url = 'https://openrouter.ai/api/v1/models'
            response = requests.get(url)
            models = json.loads(response.text)['data']   

            cls.put(path, models)

        if search != None:
            models =  [m for m in models if search in m['id']]
        return models
    
    @classmethod
    def model_names(cls, search:str = None, update=False):
        return [m['id'] for m in cls.models(search=search, update=update)]
    

    def num_tokens(self, text):
        return len(str(text).split(' '))


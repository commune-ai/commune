import commune as c
import requests
import json
import os
from typing import List

class OpenRouterModule(c.Module):
    whitelist = ['generate', 'models']

    def __init__(self,
                url:str = "https://openrouter.ai/api/v1/chat/completions",
                model: str = "openai/gpt-4o",
                role: str = "user",
                http_referer: str = "http://localhost:3000",
                api_key: str = 'OPEN_ROUTER_API_KEY',
                x_title: str = "Communne",
                max_history: int = 100,
                search: str = None,
                **kwargs
                ):
        self.url = url
        self.set_api_key(api_key)
        self.set_model(model)

        self.model = model
        self.role = role
        self.http_referer = http_referer
        self.x_title = x_title
        self.max_history = max_history


    def set_model(self, model:str):
        self.model = model
        return {"status": "success", "model": model, "models": self.models}
        
    def talk(self, *text:str, **kwargs):
        text = ' '.join(list(map(str, text)))
        return self.forward(text, **kwargs)
    ask = talk

    def forward(self, text: str, text_only:bool = True, stream = False,  model=None, history=None, max_tokens=4000, **kwargs ):
        
        model = model or self.model
        history = history or []

        c.print(f"Generating response with {model}...", color='yellow')

        data = {
                "model": model, 
                'streaming': stream,
                "messages": history + [{"role": self.role, "content": text} ],
                'max_tokens': max_tokens,
                **kwargs
            }

        response = requests.post(
            url=self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.http_referer, 
                "X-Title": self.x_title, 
    
            },

            data=json.dumps(data)
            )

        response = json.loads(response.text)
        if 'choices' not in response:
            return response
        output_text = response["choices"][0]["message"]["content"]

        if text_only:
            return output_text
        
        return response
    
    prompt = generate = forward

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
            model = self.model
        response = self.prompt(text, model=model, text_only=True)
        if isinstance(response, dict) and 'error' in response:
            return response
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
    def filter_models(cls, models, search:str = None):
        if search == None:
            return models
        if ',' in search:
            search = [s.strip() for s in search.split(',')]
        else:
            search = [search]
        models = [m for m in models if any([s in m['id'] for s in search])]
        return [m for m in models]
    
    @classmethod
    def models(cls, search:str = None, names=True, path='models', max_age=1000, update=False):
        models = cls.get(path, None , max_age=max_age, update=update)
        if models == None:
            c.print('Updating models...', color='yellow')
            url = 'https://openrouter.ai/api/v1/models'
            response = requests.get(url)
            models = json.loads(response.text)['data']  
            cls.put(path, models)
        models = cls.filter_models(models, search=search)
        if names:
            models = [m['id'] for m in models]
        return models
    
    @classmethod
    def model_names(cls, search:str = None):
        return [m['id'] for m in cls.models(search=search)]
    

    def num_tokens(self, text):
        return len(str(text).split(' '))
    

    def flocktalk(self, *text, 
                  search=None,
                   history=None, 
                   max_tokens=4000, 
                   timeout=10,
                   **kwargs):
        models = self.models(search=search)
        future2model = {}
        for model in models:
            kwargs = dict(text=' '.join(text), model=model, history=history, max_tokens=max_tokens)
            f = c.submit(self.forward,  kwargs=kwargs, timeout=timeout)
            future2model[f] = model
        model2response = {}
        futures = list(future2model.keys())
        try:
            for f in c.as_completed(futures, timeout=timeout):
                model = future2model[f]
                model2response[model] = f.result()
                c.print(f"{model}", color='yellow')
                c.print(f"{model2response[model]}", color='green')
        except Exception as e:
            c.print(f"Error: {e}", color='red')
        return model2response
    




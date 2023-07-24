
import asyncio
import commune as c

class FreeGPT(c.Module):
    def __init__(self, config = None,  **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)
        self.models = self.config.models


    model_cache = {}
    def get_model(self, model:str = None):

        if model == None:
            model = self.random_model()
        if model in self.model_cache:
            return self.model_cache[model]
        models = self.models
        freegpt = self.get_freegpt()
        assert model in models, f"Model {model} not found in {models}"
        model =  getattr(freegpt, model)   
        self.model_cache[model] = model
        return model

    async def async_forward(self, prompt:str, model=None):
        try:
            model = self.get_model(model)
            resp = await model.Completion.create(prompt)
            assert isinstance(resp, str), f"Response must be a string, not {type(resp)}"
            assert len(resp) > 0, f"Response must be a non-empty string"
            return resp
        except Exception as e:
            return {'error': str(e)}
        



    def forward(self, 
                prompt:str,
                timeout=6,
                trials=4):
        try:
            loop  = c.get_event_loop()
            jobs = [self.async_forward(prompt, model=m) for m in self.models ]
            responses = c.gather(jobs, timeout=timeout)


            for i,m in enumerate(self.models):
                resp  = responses[i]
                c.print(f'{m} -> {resp}')
                if isinstance(resp, str ) and len(resp) > 0:
                    return resp
                else:
                    c.print(f'error generating text, retrying -> {trials} left...')

        except Exception as e:
            if trials > 0:
                resp = self.forward(prompt, trials=trials-1)
            else:
                resp = self.config.default_response 
            
        return resp
    
    def random_model(self):
        return c.choice(self.models)

    @classmethod
    def available_models(cls):
        return [l.split('/')[-1].split('.')[0] for l in c.ls(cls.dirpath() + '/freeGPT') if l.endswith('.py') and '__init__' not in l]

    chat = talk = forward

    @classmethod
    def get_freegpt(cls):
        try:
            import freeGPT
        except ModuleNotFoundError:
            c.cmd(f'pip install -e {cls.dirpath()}')
            import freeGPT

        return freeGPT
        
            


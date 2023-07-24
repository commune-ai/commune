
import asyncio
import commune as c

class freegpt(c.Module):
    def __init__(self, config = None,  **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)


    def set_model(self, model):
        models = self.models()
        freegpt = self.get_freegpt()
        assert model in models, f"Model {model} not found in {models}"
        self.model = getattr(freegpt, model)   

    async def async_forward(self, prompt:str,):
        try:
            resp = await self.model.Completion.create(prompt)
            return resp
        except Exception as e:
            return str(e)

    def forward(self, prompt:str,):
        loop  = c.get_event_loop()
        resp = loop.run_until_complete(self.async_forward(prompt))
        return resp
    @classmethod
    def models(cls):
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
        
            


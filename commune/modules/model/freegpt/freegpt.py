
import asyncio
import commune as c

class freegpt(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        import freeGPT
        self.model = getattr(freeGPT, "gpt3")   


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

    chat = talk = forward

    @classmethod
    def install(cls):
        c.pip_install('freeGPT')

            


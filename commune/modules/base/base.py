import commune as c

class Base(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
    def run(self):
        print('Base run')


"""
import freeGPT
import asyncio

def chat(prompt):
    async def main():
        try:
            resp = await getattr(freeGPT, "gpt3").Completion.create(prompt)
            return resp
        except Exception as e:
            return str(e)
    return asyncio.get_event_loop().run_until_complete(main())

print(chat("hey!"))
"""


# Write a module based on the Base Module 
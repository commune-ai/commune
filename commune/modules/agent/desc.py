    
class Desc:

    def __init__(self, agent='agent'):
        self.agent = c.module(agent)()
        
    def forward(self, module, max_age=0):
        code= c.code(module)
        code_hash = c.hash(code)
        path = self.resolve_path(f'summary/{module}.json')
        output = c.get(path, max_age=max_age)
        if output != None:
            return output

        prompt = {
                "task": "summarize the following into tupples and make sure you compress as much as oyu can",
                "code": code,
                "hash": code_hash,
                }
        output = ''
        for ch in self.agent.forward(str(prompt), preprocess=False):
            output += ch
            print(ch, end='')
        c.put(path, output)
        print('Writing to path -->', path)
        return output

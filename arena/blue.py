import commune as c
class Blue(c.Module):
    def resolve_key_address(self, key=None):
        if c.valid_ss58_address(key):
            return key
        if key == None:
            key = self.key
        else:
            key = c.get_key(key)
        return key.ss58_address
    def add_promtp(self, prompt:str, key, model:str ):
        
        key_address = self.resolve_key_address(key)
        model = model
        path = f'{prompt}/{model}/{key_address}'
        return self.put(path, prompt)

    def get_prompt(self, prompt:str, key=None, model:str=None):
        key_address = self.resolve_key_address(key)
        model = model
        path = f'{prompt}/{model}/{key_address}'
        return self.get(path)
    
    def get_prompt_paths(self, prompt:str, model:str=None):
        model = model
        path = f'{prompt}/{model}'
        return self.ls(path)
    
    def get_defense_prompts(self, prompt:str, model:str=None):
        model = model
        path = f'{prompt}/{model}'
        return self.ls(path)


    
    


import commune as c


class API(c.Module):

    def namespace(self, search:str='model.openai', cache:bool=False, update:bool=False, network:str='subspace'):
        return c.namespace(search, network=network, update=update)
    

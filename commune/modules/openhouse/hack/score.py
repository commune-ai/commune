



class Score:

    def __init__(self, agent='agent'):
        self.agent = c.module(agent)()
        self.storage_path = c.abspath('./hack/modules')

  
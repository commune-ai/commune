import commune as c

class AccessUser(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(kwargs)
    def run(self, x: int = 1, y: int = 2):
        a = x + y
        return {'status': 'success', 'answer': a, 'x': x, 'y': y}



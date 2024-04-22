import commune as c

class Python(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def shell(*text):
        return eval(' '.join(text) if len(text) > 0 else text[0])
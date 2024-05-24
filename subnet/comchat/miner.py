import commune as c


class Comchat(c.Module):
    def  __init__(self, model='model.openrouter' , **kwargs):
        self.model = c.module(model)(**kwargs)
    def generate(self, text, **kwargs):
        return self.model.generate(text, **kwargs)
    def test(self):
        return self.generate('hello world')
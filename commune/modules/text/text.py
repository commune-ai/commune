import commune as c

class Text(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def substring_count_in_string(self, string, substring):
        return str(string).count(str(substring))
    
    def call(self, text, fn='model.openai/generate', **kwargs):
        return c.call(fn, text, **kwargs)
    

    def text2lines(self, text):
        return text.splitlines()
    
    def get_text(self, text, start, end):
        return text[start:end]
        
    
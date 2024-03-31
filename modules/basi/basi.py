import commune as c

class Basi(c.Module):
    prompt = """
    BASI PROMPT
    """
    def __init__(self, a=1, b=2):
        self.model = c.module('model.openrouter')()

    def call(self, *text) -> int:
        text = ' '.join(text)
        prompt = f"""
        {self.prompt}
        {text}
        """
        return self.model.forward(prompt)
    
        
    
import commune as c

class Docs(c.Module):

    def ask(self, *question, model='anthropic/claude-3.5-sonnet', **kwargs ) -> int:
        question = ' '.join(list(map(str, question)))
        prompt = f"""
        Question:
            {question}
        Context:
          
        """
        return c.module('chat')().generate(prompt, model=model,stream=1, **kwargs)
    
    
    def get_context(self):
        return c.file2text(self.dirpath())
import commune as c

class Docs(c.Module):
    def forward(self, *question, 
                model='anthropic/claude-3.5-sonnet', 
                **kwargs ) -> int:
        question = ' '.join(list(map(str, question)))
        context = c.file2text(self.dirpath())
        prompt = f"""
        QUESTION:
            {question}
        CONTEXT:
            {context}
        """
        return c.module('chat')().generate(prompt, model=model,stream=1, **kwargs)
    ask = forward
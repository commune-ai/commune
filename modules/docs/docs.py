import commune as c
import os
class Docs:
    def forward(self, *question, 
                model='anthropic/claude-3.5-sonnet', 
                **kwargs ) -> int:
        question = ' '.join(list(map(str, question)))
        context = c.file2text(os.path.dirname(__file__) )
        prompt = f"""
        QUESTION:
            {question}
        CONTEXT:
            {context}
        """
        return c.ask(prompt, model=model, **kwargs)
    
    
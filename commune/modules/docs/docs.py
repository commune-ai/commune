import commune as c
import os
class Docs:
    def forward(self, module='module', 
                model='anthropic/claude-3.5-sonnet', 
                fmt='DICT(key=topic:str, value=data:str)', 
                goal=" summarize the following into object level descriptions get the core functions of the modules and give some examples", 

    **kwargs ) -> int:
        context = c.code(module)
        prompt = f"""
        GOAL:
            {goal}
        CONTEXT:
            {context}
        OUTPUT:

        """
        output = ''
        for ch in c.ask(prompt, model=model, process_input=False, **kwargs):
            print(ch, end='')
            output += ch
        return output   

    
    
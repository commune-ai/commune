import commune as c

class Readme:
    def forward(self, module='vali', max_age=1, update=False):
        code = c.code(module)
        path = c.dirpath(module) + '/README.md'
        readme = c.get('readme',None, max_age=max_age, update=update)
        if readme:
            c.print(f"README file already exists for module {module}", color='yellow')
            return readme



        prompt = f'''
        Generate a README file for a Python module.
        CODE: {code}
        CONTEXT: {c.core_context()}
        OUTPUT_FORMAT: 
        <START_OUTPUT>
        text
        <END_OUTPUT>
        08
        '''
        response =  c.ask(prompt, process_input=False)
        output = ''
        for ch in response:
            print(ch, end='')
            output += str(ch)

        output = output.split('<START_OUTPUT>')[1].split('<END_OUTPUT>')[0]
        
        
        c.put_text(path, output)

        return response



    
import commune as c
import time
import os
# import agbuildent as h

class Build:
    anchor = 'OUTPUT'
    endpoints = ["build"]

    def __init__(self, 
                 model = None,
                 key = None,
                **kwargs):
        
        self.model = c.module('model.openrouter')(model=model)
        self.models = self.model.models()
        self.key = c.get_key(key)

    def process_text(self, text):
        for ch in text.split(' '):

            if len(ch) > 0 and ch[0] in ['.', '/', '~'] and os.path.exists(ch):
                text = text.replace(ch, str(c.file2text(ch)))
        return text
    
    def forward(self, 
                 text, 
                 *extra_text, 
                 temperature= 0.5, 
                 max_tokens= 1000000, 
                 model= 'anthropic/claude-3.5-sonnet', 
                 path = None,
                 simple = False,
                 stream=True
                 ):
        
        advanced_mode =  """
        - Please use  to name the repository and
        - This is a a full repository construction and please
        - INCLUDE A README.md AND a scripts folder with the build.sh 
        - file to build hte environment in docker and a run.sh file 
        - to run the environment in docker
        - INCLUDE A TESTS folder for pytest
        """ 

        task =  f"""
            YOU ARE A CODER, YOU ARE MR.ROBOT, YOU ARE TRYING TO BUILD IN A SIMPLE
            LEONARDO DA VINCI WAY, YOU ARE A agent, YOU ARE A GENIUS, YOU ARE A STAR, 
            YOU FINISH ALL OF YOUR REQUESTS WITH UTMOST PRECISION AND SPEED, YOU WILL ALWAYS 
            MAKE SURE THIS WORKS TO MAKE ANYONE CODE. YOU HAVE THE CONTEXT AND INPUTS FOR ASSISTANCE
            {advanced_mode if not simple else ''}
            """

        prompt = f"""
            -- TASK --
            {task}
            -- OUTPUT FORMAT --
            <{self.anchor}(path/to/file)> # start of file
            FILE CONTENT
            </{self.anchor}(path/to/file)> # end of file
            -- OUTPUT --
            """
        if len(extra_text) > 0:
            text = ' '.join(list(map(str, [text] +list(extra_text))))

        text = self.process_text(text)
        output =  self.model.generate( prompt + '\n' + text,
                                      stream=stream, 
                                      model=model, 
                                      max_tokens=max_tokens, 
                                      temperature=temperature )
        if path == None:
            return output
        return self.process_output(output, path=path)
    
    def process_output(self, response, path=None):
        # generator = self.search_output('app')
        if path == None:
            return response
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        assert os.path.exists(path), f'Path does not exist: {path}'
        path = os.path.abspath(path)
        buffer = '-------------'
        anchors = [f'<{self.anchor}(', f'</{self.anchor}(']
        color = c.random_color()
        content = ''
        for token in response:
            content += str(token)
            is_file_in_content =  content.count(anchors[0]) == 1 and content.count(anchors[1]) == 1
            c.print(token, end='', color=color)
            if is_file_in_content:
                file_path = content.split(anchors[0])[1].split(')>')[0] 
                file_content = content.split(anchors[0] +file_path + ')>')[1].split(anchors[1])[0] 
                c.put_text(path + '/' + file_path, file_content)
                c.print(buffer,'Writing file --> ', file_path, buffer, color=color)
                content = ''
                color = c.random_color()
        return {'path': path, 'msg': 'File written successfully'}
    
    def prompt_args(self):
        # get all of the names of the variables in the prompt
        prompt = self.prompt
        variables = []
        for line in prompt.split('\n'):
            if '{' in line and '}' in line:
                variable = line.split('{')[1].split('}')[0]
                variables.append(variable)
        return list(set(variables))
    
    def utils_path(self):
        return os.path.dirname(__file__) + '/utils.py'

    def utils(self):
        return c.find_functions(self.utils_path())
import commune as c
import time
import os
# import agbuildent as h

class Edit:
    anchors = ['<START_OUTPUT(', '<END_OUTPUT(']
    endpoints = ["build"]

    def __init__(self, 
                 model = None,
                 key = None,
                **kwargs):

        self.model = c.module('agent')(model=model)
        self.models = self.model.models()
        self.key = c.get_key(key)

    def find_files(self, query:str='the file that is the core of this folder', path:str='./', model:str='anthropic/claude-3.5-sonnet-20240620:beta', n:int=30):
        file2content = {}
        for p in c.fn('find/files', params=dict(query=query, path=path, model=model, n=n)):
            try:
                file2content[p] = c.get_text(p)
            except Exception as e:
                continue
            
        return file2content

    def forward(self,
                text = 'edit the file',
                *extra_text,
                path = './', 
                 task = None,
                 temperature= 0.5, 
                 module=None,
                 max_tokens= 1000000, 
                 threshold= 1000000,
                 model= 'anthropic/claude-3.5-sonnet', 
                 write=False,
                 stream=True
                 ):
        text = text + ' ' + ' '.join(extra_text)
        if module:
            path = c.filepath(module)

        task =  task or f"""

            YOU ARE GIVEN A MAP OF FILES AND NEED TO REPAIR BASED ON THE CONTEXT
            YOU CAN SUGGEST CHANGES TO FILES AND SUGGEST DELETIONS OF EXISTING FILES
            YOU ARE A CODER AND EDITOR, YOU ARE MR.ROBOT, YOU ARE TRYING TO BUILD IN A SIMPLE
            LEONARDO DA VINCI WAY, YOU ARE A agent, YOU ARE A GENIUS, YOU ARE A STAR, 
            YOU FINISH ALL OF YOUR REQUESTS WITH UTMOST PRECISION AND SPEED, YOU WILL ALWAYS 
            MAKE SURE THIS WORKS TO MAKE ANYONE CODE. YOU HAVE THE CONTEXT AND INPUTS FOR ASSISTANCE
            - include all of the file content and dont HALF ASS, FULL ASS   
            - RETURN ALL OF THE FILE PATHS AS IM RECONSTRUCTING IT
            """

        context = c.file2text(path)
        if len(str(context)) > threshold :
            print('Finding Relevant Files')
            context = self.find_files(query=task, path=path)
        prompt = f"""
            -- TASK --
            {task}
            -- CONTEXT --x
            {context}
            {text if text else ''}
            FILE CONTENT
            -- FORMAT --
            {self.anchors[0]}(path/to/file)> # start of file
            FILE CONTENT
            {self.anchors[1]}(path/to/file)> # end of file

            IF YOU WANT TO DELETE A FILE USE THE FOLLOWING FORMAT
            <DELETE(path/to/fileorfolder)>

            YOU CAN DO IT
            -- OUTPUT --
        """
        output =  self.model.generate(prompt, stream=stream, model=model, max_tokens=max_tokens, temperature=temperature , process_text=False)
        return self.process_output(output, path=path, write=write)
    
    def process_output(self, response, path=None, write=False):
        if path == None:
            return response
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        assert os.path.exists(path), f'Path does not exist: {path}'
        path = os.path.abspath(path)
        buffer = '-------------'
        color = c.random_color()
        content = ''
        file2output = {}
        for token in response:
            content += str(token)
            is_file_in_content =  content.count(self.anchors[0]) == 1 and content.count(self.anchors[1]) == 1
            c.print(token, end='', color=color)
            if is_file_in_content:
                file_path = content.split(self.anchors[0])[1].split(')>')[0] 
                file_content = content.split(self.anchors[0] +file_path + ')>')[1].split(self.anchors[1])[0] 
                if write:
                    print(f'Writing file: {file_path}')
                    c.print(c.put_text(file_path, file_content))
                c.print(f'{buffer}Writing file --> {file_path}{buffer}', color=color)
                content = ''
                file2output[file_path] = file_content
                color = c.random_color()
        return {'path': path, 'msg': 'File written successfully', 'file2output': file2output, 'write': write}
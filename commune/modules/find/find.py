



import commune as c
import json
import os

class Find:
    model='anthropic/claude-3.5-sonnet-20240620:beta'

    def forward(self,  
              query='most relevant modules', 
              options: list[str] = [],  
              n=10,  
              threshold=0.5,
              model='anthropic/claude-3.5-sonnet-20240620:beta'):

        front_anchor = f"<OUTPUT>"
        back_anchor = f"</OUTPUT>"
        idx2options = {i:option for i, option in enumerate(options)}
        home_path = c.resolve_path('~')
        for idx, option in idx2options.items():
            if option.startswith(home_path):
                idx2options[idx] = option.replace(home_path, '~/')

        print(f"Querying {query} with options {options}")
        prompt = f"""
        QUERY
        {query}
        OPTIONS 
        {idx2options} 
        INSTRUCTION 
        only output the IDX:int  and score OF AT MOST {n} BUT YOU DONT NEED TO FOR SIMPLICITY
        OUTPUT
        (JSON ONLY AND ONLY RESPOND WITH THE FOLLOWING INCLUDING THE ANCHORS SO WE CAN PARSE) 
        <OUTPUT>DICT(data:list[[idx:int, score:float]])</OUTPUT>
        """
        output = ''
        for ch in c.ask(prompt, model=model): 
            print(ch, end='')
            output += ch
            if ch == front_anchor:
                break
        if '```json' in output:
            output = output.split('```json')[1].split('```')[0]
        elif front_anchor in output:
            output = output.split(front_anchor)[1].split(back_anchor)[0]
        else:
            output = output
        output = json.loads(output)
        assert len(output) > 0
        output = [options[idx] for idx, score in output["data"]  if len(options) > idx and score > threshold]
        return output


    def file2text(self, path):
        file2text = {}
        for file in self.files(path=path):
            file2text[file] = c.get_text(file)
        return file2text
        
    @classmethod
    def lines(self,  search:str=None, path:str='./') -> list[str]:
        """
        Finds the lines in text with search
        """
        # if is a directory, get all files
        file2lines = {}
        for file, text in c.file2text(path).items():
            found_lines = []
            lines = text.split('\n')
            idx2line = {idx:line for idx, line in enumerate(lines)}
            for idx, line in idx2line.items():
                if search in line:
                    found_lines.append((idx, line))
            file2lines[file] = found_lines
        return file2lines

    def files(self,
              query='the file that is the core of this folder',
               path='./',  
               model='anthropic/claude-3.5-sonnet-20240620:beta', 
               n=30):
        files =  c.files(path)
        homepath = c.resolve_path('~/')
        for i, file in enumerate(files):
            if file.startswith(homepath):
                files[i] = file.replace(homepath, '~/')
        files =  self.forward(options=files, query=query, n=n, model=model)
        return [c.abspath(path+k) for k in files]

    def modules(self,  query='the filel that is the core of commune', model='anthropic/claude-3.5-sonnet-20240620:beta'): 
        return self.forward(options=c.get_modules(), query=query, model=model)

    def utils(self, query='confuse the gradients', model='anthropic/claude-3.5-sonnet-20240620:beta'):
        return self.forward(query=query, options=c.get_utils(), model=model)
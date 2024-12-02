import commune as c
import json
import os

class Search:
    description = "This module is used to find files and modules in the current directory"
    def forward(self, query='', mode='modules'):
        return getattr(self, f'find_{mode}')(query=query)
    

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
    
    def query(self,  options,  
              query='most relevant modules', 
              output_format="DICT(data:list[[key:str, score:float]])",  
              path='./', 
              anchor = 'OUTPUT', 
              n=10,  
              model='sonnet'):

        front_anchor = f"<{anchor}>"
        back_anchor = f"</{anchor}>"
        print(f"Querying {query} with options {options}")
        prompt = f"""
        QUERY
        {query}
        OPTIONS 
        {options} 
        INSTRUCTION 
        get the top {n} functions that match the query
        OUTPUT
        (JSON ONLY AND ONLY RESPOND WITH THE FOLLOWING INCLUDING THE ANCHORS SO WE CAN PARSE) 
        {front_anchor}{output_format}{back_anchor}
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
        return output

    def files(self, query='the file that is the core of commune',  path='./',  model='sonnet' ):
        return self.query(options=c.files(path), query=query)

    def modules(self,  query='the filel that is the core of commune',  model='sonnet'): 
        return self.query(options=c.modules(), query=query)

    def utils(self, query='confuse the gradients'):
        return self.query(query=query, options=c.get_utils())

import commune as c
import json
import os

class Summary:
    description = "This module is used to find files and modules in the current directory"
    def forward(self, path='./', **kwargs):
        return self.reduce(str(self.file2text(path=path), **kwargs))

    def reduce(self, text, max_chars=10000 , timeout=40, max_age=30, model='anthropic/claude-3.5-sonnet-20240620:beta'):
        if os.path.exists(text): 
            path = text
            if os.path.isdir(path):
                print('REDUCING A DIRECTORY -->', path)
                future2path = {}
                path2result = {}
                paths = c.files(path)
                progress = c.tqdm(len(paths), desc='Reducing', leave=False)
                while len(paths) > 0:
                    for p in paths:
                        future = c.submit(self.reduce, [p], timeout=timeout)
                        future2path[future] = p
                    try:
                        for future in c.as_completed(future2path, timeout=timeout):
                            p = future2path[future]
                            r = future.result()
                            paths.remove(p)
                            path2result[p] = r
                            print('REDUCING A FILE -->', r)
                            progress.update(1)
                    except Exception as e:
                        print(e)
                return path2result
            else:
                assert os.path.exists(path), f'Path {path} does not exist'
                print('REDUCING A FILE -->', path)
                text = str(c.get_text(path))
        elif c.module_exists(text):
            text = c.code(text)

        code_hash = c.hash(text)
        path = f'summary/{code_hash}' 

        text = f'''
        GOAL
        summarize the following into tupples and make sure you compress as much as oyu can
        CONTEXT
        {text}
        OUTPUT FORMAT ONLY BETWEEN THE TAGS SO WE CAN PARSE
        <OUTPUT>DICT(data=List[str])</OUTPUT>
        '''
        if len(text) >= max_chars * 2 :
            batch_text = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
            for i, t in enumerate(batch_text):
                result = c.ask(t, model=model, stream=0)
                if i == 0:
                    result = result.split('<OUTPUT>')[1]
                if i == len(batch_text) - 1:
                    result = result.split('</OUTPUT>')[0]
                path2result[path] = result
            return result
        if "'''" in text:
            text = text.replace("'''", '"""')
        data =  c.ask(text, model=model, stream=0)

        return {"data": self.process_data(data)}

    def process_data(self, data):
        try:
            data = data.split('<OUTPUT>')[1].split('</OUTPUT>')[0]
            return data
        except:
            return data

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
    
    def query(self,  options,  
              query='most relevant modules', 
              output_format="DICT(data:list[[idx:int, score:float]])",  
              path='./', 
              anchor = 'OUTPUT', 
              n=10,  
              model='anthropic/claude-3.5-sonnet-20240620:beta'):

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
        output = json.loads(output)["data"]
        output = [int(k) for k,v in output, if ]
        output = [options[k] for k in output ]
        assert len(output) > 0
        return output

    def query_files(self,  path='./', query='return only core path indexes',  n=10):
        files =  self.query(options=c.files(path), query=query, n=n)
        return [c.abspath(path+k) for k in files]

    def modules(self,  query='the filel that is the core of commune',  model='sonnet'): 
        return self.query(options=c.modules(), query=query)

    def utils(self, query='confuse the gradients'):
        return self.query(query=query, options=c.get_utils())

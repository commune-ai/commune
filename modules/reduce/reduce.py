-



import commune as c
import json
import os


class Reduce:
    description = "This module is used to find files and modules in the current directory"
    model='anthropic/claude-3.5-sonnet-20240620:beta'
    def forward(self,  text, model=model,  timeout=10, n=10):
        if isinstance(text, str) and os.path.exists(text): 
            file2text = c.file2text(text)
            futures = []
            future2file = {}
            file2summary = c.file2text(text)
            for i, (file, text) in enumerate(file2text.items()):
                future = c.submit(self.forward, dict(text=text, model=model), timeout=timeout)
                future2file[future] = file
            progress = c.progress(len(futures), total=len(file2text))

            try:
                for future in c.as_completed(future2file, timeout=timeout):
                    file = future2file[future]
                    file2summary[file] = future.result()
                    progress.update()
  
            except Exception as e:
                print(e)
                pass
            return file2summary
            
        text = f'''
        GOAL
        summarize the following into tupples and make sure you compress as much as oyu can
        CONTEXT
        {text}
        OUTPUT FORMAT IN JSON FORMAT ONLY BETWEEN THE ANCHORS ONLY INCLUDE PURE JSON, NO TEXT
        <OUTPUT>JSON(data:list[])</OUTPUT>
        '''
        assert len(text) < 20000
        return self.process_data(c.ask(text, model=model, stream=0))

    def process_data(self, data):
        try:
            data = data.split('<OUTPUT>')[1].split('</OUTPUT>')[0]
            data = json.loads(data)
            return data
        except:
            return c

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
    
    def forward(self,  options,  
              query='most relevant information', 
              output_format="DICT(data:list[[idx:str, score:float]])",  
              anchor = 'OUTPUT', 
              n=10,  
              model='anthropic/claude-3.5-sonnet-20240620:beta'):

        front_anchor = f"<{anchor}>"
        back_anchor = f"</{anchor}>"
        idx2options = {i:option for i, option in enumerate(options)}
        print(f"Querying {query} with options {options}")
        prompt = f"""
        QUERY
        {query}
        OPTIONS 
        {idx2options} 
        INSTRUCTION 
        only output the IDX  and score of the TOP {n} FUNCTIONS that match the query
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
        output_idx_list =  [int(k) for k,v in output["data"]]
        output = [options[i] for i in output_idx_list  if len(options) > i]
    
        return output

    def files(self, path='./',  query='the file that is the core of this folder',  n=30, model='anthropic/claude-3.5-sonnet-20240620:beta'):
        files = c.files(path)
        home = os.path.expanduser('~')
        files = [file.replace(home, '~') for file in files if home in file]
        files =  self.query(options=files, query=query, n=n, model=model)
        return files

    def modules(self,  query='the filel that is the core of commune', model='anthropic/claude-3.5-sonnet-20240620:beta'): 
        return self.query(options=c.modules(), query=query, model=model)

    def utils(self, query='confuse the gradients', model='anthropic/claude-3.5-sonnet-20240620:beta'):
        return self.query(query=query, options=c.get_utils(), model=model)

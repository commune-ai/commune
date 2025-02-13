



import commune as c
import json
import os

class Find:
    model='google/gemini-2.0-flash-001'

    def forward(self,  
              options: list[str] = [],  
              query='most relevant', 
              n=10,  
              trials = 3,
              min_score=0,
              max_score=100,
              threshold=90,
              context = None,
              model=None):

        model = model or self.model
        if trials > 0 :
            try:
                return self.forward(options=options, query=query, n=n, trials=trials-1, threshold=threshold, context=context, model=model)
            except Exception as e:
                print(e)
                if trials == 0:
                    raise e
                else:
                    return self.forward(options=options, query=query,  n=n, trials=trials-1, threshold=threshold, context=context, model=model)
        anchors = [f"<START_JSON>", f"</END_JSON>"]
        if isinstance(options, dict):
            options  = list(options.keys())
        idx2options = {i:option for i, option in enumerate(options)}
        prompt = f"""
        --QUERY--
        {query}
        CONTEXT{context}
        --OPTIONS--
        {idx2options} 
        --RULES--
        only output the IDX:int  and score OF AT MOST {n}
        BUT YOU DONT NEED TO FOR SIMPLICITY TO NOT RETURN WITH COMMENTS
        MIN_SCORE:{min_score}
        MAX_SCORE:{max_score}
        THRESHOLD:{threshold}
        DO NOT RETURN THE OPTIONS THAT SCORE BELOW THRESHOLD({threshold})
        BE CONSERVATIVE WITH YOUR SCORING TO SAVE TIME
        THE MINIMUM SCORE IS 0 AND THE MAXIMUM SCORE IS 100
        --OUTPUT_FORMAT--
        (RETURN ONLY IN JSON FORMAT SO IT CAN BE PARSED EASILY BETWEEN THE ANCHORS, RESPOND IN FULL JSON FORMAT)
        {anchors[0]}DICT(data:LIST(idx:INT, score:INT)]){anchors[1]}
        --OUTPUT--
        '''json
        """
        output = ''
        for ch in c.ask(prompt, model=model): 
            print(ch, end='')
            output += ch
            if ch == anchors[1]:
                break
        if '```json' in output:
            output = output.split('```json')[1].split('```')[0]
        elif anchors[0] in output:
            output = output.split(anchors[0])[1].split(anchors[1])[0]
        else:
            output = output
        output = json.loads(output)
        assert len(output) > 0
        output = [options[idx] for  idx, score in output["data"]  if len(options) > idx and score > threshold]
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
               model='google/gemini-2.0-flash-001', 
               n=30):
        model = model or self.model
        files =  c.files(path)
        files =  self.forward(options=files, query=query, n=n, model=model)
        return files
        return [c.abspath(k) for k in files]

    def modules(self,  query='', **kwargs): 
        return self.forward(options=c.get_modules(), query=query,**kwargs)

    def utils(self, query='confuse the gradients', model='anthropic/claude-3.5-sonnet-20240620:beta'):
        return self.forward(query=query, options=c.get_utils(), model=model)

    
    def fn(self, query:str='something that i can find functions in', *extra_query, module2fns = None):
        query =' '.join([query] +list(extra_query))
        if module2fns is None:
            module2fns = c.module2fns()
        options = []
        for module, fns in module2fns.items():
            for fn in fns:
                options += [f"{module}/{fn}"]
        context  = f'''
        {c.text(c.docs_path)}
        '''
        
        return self.forward(query=query, options=options)

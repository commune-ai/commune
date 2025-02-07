



import commune as c
import json
import os

class Find:
    def forward(self,  
              options: list[str] = [],  
              query='most relevant', 
              rules = '''''',
               output_format="DICT(data:list[[idx:int, score:float]])",
              n=10,  
              trials = 3,
              threshold=0.5,
              context = None,
              model='anthropic/claude-3.5-sonnet-20240620:beta'):


        if trials > 0 :
            try:
                return self.forward(options=options, query=query, rules=rules, output_format=output_format, n=n, trials=trials-1, threshold=threshold, context=context, model=model)
            except Exception as e:
                print(e)
                if trials == 0:
                    raise e
                else:
                    return self.forward(options=options, query=query, rules=rules, output_format=output_format, n=n, trials=trials-1, threshold=threshold, context=context, model=model)


        anchors = [f"<START_JSON_RESULT>", f"<END_JSON_RESULT>"]

        if isinstance(options, dict):
            options  = list(options.keys())
        idx2options = {i:option for i, option in enumerate(options)}
        print(f"Querying {query} options={len(options)}")
        prompt = f"""
        QUERY
        {query}
        CONTEXT
        {context}
        OPTIONS 
        {idx2options} 
        RULES 
        only output the IDX:int  and score OF AT MOST {n}
        BUT YOU DONT NEED TO FOR SIMPLICITY TO NOT RETURN WITH COMMENTS
        OUTPUT
        (JSON ONLY AND ONLY RESPOND WITH THE FOLLOWING INCLUDING THE ANCHORS SO WE CAN PARSE) 
        {anchors[0]}{output_format}{anchors[1]}
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
        print(output, 'FAM')
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
        files =  self.forward(options=files, query=query, n=n, model=model)
        return [c.abspath(k) for k in files]

    def modules(self,  query='', model='anthropic/claude-3.5-sonnet-20240620:beta'): 
        module2fns = []
        return self.forward(options=c.get_modules(), query=query, model=model, context=c.module2fns())

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

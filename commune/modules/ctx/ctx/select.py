
import commune as c
import json
import os



class Selector:

    def forward(self,  
              options: list[str] = [],  
              query='most relevant', 
              n=10,  
              trials = 3,
              min_score=0,
              max_score=9,
              threshold=5,
              model='anthropic/claude-3.5-sonnet',
              context = None):

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
        THE MINIMUM SCORE IS 0 AND THE MAXIMUM SCORE IS 10
        --OUTPUT_FORMAT--
        {anchors[0]}DICT(data:LIST[LIST[idx:INT, score:INT]]]){anchors[1]}
        MAKE SURE YOU RETURN IT THE JSON FORMAT BETWEEN THE ANCHORS AND NOTHING ELSE TO FUCK UP 
        --OUTPUT--
        """
        output = ''
        for ch in c.ask(prompt, model=model): 
            print(ch, end='')
            output += ch
            if ch == anchors[1]:
                break
        if anchors[0] in output:
            output = output.split(anchors[0])[1].split(anchors[1])[0]
        else:
            output = output
        print(output)
        output = json.loads(output)
        assert len(output) > 0
        output = [idx2options[idx] for idx, score in output['data'] if score >= threshold]
        return output

    def files(self,
                query='the most relevant files',
                *extra_query,
               path='./',  
               n=30):
        if len(extra_query)>0:
            query = ' '.join([query, *extra_query])
        options = self.forward(options=c.files(path), query=query, n=n)
        return options

    def modules(self,  query='', **kwargs): 
        return self.forward(options=c.mods(), query=query,**kwargs)

    def utils(self, query='i want something that does ls', **kwargs):
        return self.forward(query=query, options=c.get_utils(), **kwargs)
    
    def search(self, query:str='how can i stake on chain', **kwargs):
        module2schema = c.module2schema()
        options = []
        for module, schema in module2schema.items():
            for fn in schema.keys():
                options += [f"{module}/{fn}"]
        context  = f'''
        '''
        return self.forward(query=query, options=options)

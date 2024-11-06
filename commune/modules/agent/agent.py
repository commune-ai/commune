import commune as c
import json
import os
class Agent:
    anchor="OUTPUT"

    def ask(self, *args, path='./'):
        text = self.args2text(args)
        context = self.summarize(query=text, path=path)
        prompt = f"""
        {context}
        AD START FINISH THE OUTPUT WITH THE ANCHOR TAGS
        if you write a file so i can easily process it back
        <{self.anchor}(path=filepath)></{self.anchor}(path=wherethefilepathwillbe)> 
        you are totally fine using ./ if you are refering to the pwd for brevity
        """
        return c.ask(prompt)

    def args2text(self, args):
        return ' '.join(list(map(str, args)))

    def get_context(self,
                        path='./', 
                        query='what are the required packages', 
                        instruction= "Which files are relevant to you?",
                        output_format="DICT(files:list[str])"):
    
        c.print('FINDING RELEVANT FILES IN THE PATH {}'.format(path), color='green')
        files = c.files(path)
        prompt = f"""
        QUERY \n {query} \n INSTRUCTION \n {instruction} \n CONTEXT {files}  {query }
        OUTPUT FORMAT
        USE THE FOLLOWING FORMAT FOR OUTPUT WITH A JSON STRING IN THE CENTER
        <{self.anchor}>{output_format})</{self.anchor}>

        """
    
        output = ''
        for ch in c.ask(prompt): 
            print(ch, end='')
            output += ch
            if ch == f'</{self.anchor}>':
                break
        
        files =  json.loads(output.split('<' +self.anchor + '>')[1].split('</'+self.anchor + '>')[0])['files']
        file2text = {c.get_text(f) for f in files}
        return file2text

    def score(self, *args, path='./'):
        text = self.args2text(args)
        context = self.get_context(text, path=path)
        return c.ask(self.prompt.format(context=context, text=text))


    def summary(self, path='./', 
                query = "get all of the important objects and a description",
                 anchor = 'OUTPUT', 
                 max_ = 100000,
                ):

        self.batch_context(path=path)
        context = c.file2text(path)
                     
        prompt = f"""
        INSTRUCTION
        SUMMARIZE the info as a black hole condenses info 
        ensure to include all necessary info and discard 
        useless info. do it as such. use the query to condition
        the tuples listed.

        [head, relation, tail]

        QUERY
        {query}
        OUTPUT FORMAT
        USE THE FOLLOWING FORMAT FOR OUTPUT WITH A JSON STRING IN THE CENTER
        store it in a relations
        <{anchor}>
        DICT(relations:list[str])
        </{anchor}>
        CONTEXT
        {context}
        """
        output = ''
        for ch in c.ask(prompt): 
            print(ch, end='')
            output += ch
            if ch == f'</{self.anchor}>':
                break
        return json.loads(output.split('<' +self.anchor + '>')[1].split('</'+self.anchor + '>')[0])


    def batch_context(self, path='./', batch_size=20000):

        file2text =  c.file2text(path)
        file2size = {k:len(v) for k,v in file2text.items()}
        current_size = 0
        batch_list = []
        files_batch = {}
        for f, s in file2size.items():
            if (current_size + s) > batch_size:
                batch_list += [files_batch]
                files_batch = {}
                current_size = 0
            current_size += s
            files_batch[f]  = c.get_text(path + f )
        return batch_list

    
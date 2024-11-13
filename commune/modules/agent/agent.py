import commune as c
import json
import os
class Agent:
    anchor="OUTPUT"
    def build(self, *args, path=c.docs_path):
        text = self.args2text(args)
        context = self.summary(query=text, path=path)
        prompt = f"""
        {context}
        AD START FINISH THE OUTPUT WITH THE ANCHOR TAGS
        if you write a file so i can easily process it back
        <{self.anchor}(path=filepath)></{self.anchor}(path=wherethefilepathwillbe)> 
        you are totally fine using ./ if you are refering to the pwd for brevity
        """
        output = ''

        front_anchor = '<OUTPUT(' 
        for ch in c.ask(prompt):
            output += output
            if front_anchor in output:
                output.split(front_anchor)[1]
            yield ch




    def args2text(self, args):
        return ' '.join(list(map(str, args)))

    def relevent_files(self,  path='./', output_format="DICT(data:list[str])" , query='',):
        front_anchor = f"<{self.anchor}>"
        back_anchor = f"</{self.anchor}>"
        instruction = "Which files are relevant to you? do not incldue the full link and use ~ if possible"
        files = c.files(path)
        prompt = f"""
        QUERY \n {query} 
        INSTRUCTION \n {instruction}
        CONTEXT \n {files}
        OUTPUT FORMAT \n (JSON ONLY AND ONLY RESPOND WITH THE FOLLOWING INCLUDING THE ANCHORS) 
        {front_anchor}{output_format}{back_anchor}
        """

        output = ''
        for ch in c.ask(prompt): 
            print(ch, end='')
            output += ch
            if ch == front_anchor:
                break
        files =  json.loads(output.split(front_anchor)[1].split(back_anchor)[0])['data']
        return files

    def summary(self, 
                path='./', 
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

    def tools(self):
        return c.fns("fn")
    
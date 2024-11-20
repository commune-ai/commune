import commune as c
import json
import os

class Agent(c.Module):
    anchor="OUTPUT"
    def build(self, *args, path=c.docs_path):
        text = self.args2text(args)
        context = self.find_text(query=text, path=path)
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
                content = output.split(front_anchor)[1]
            
    def args2text(self, args):
        return ' '.join(list(map(str, args)))
    
    def find_text(self, *args, **kwargs):
        text =  [c.get_text(f) for f in self.find_files(*args, **kwargs)]
        size = self.get_size(text)
        return size

    def get_size(self, x):
        return len(str(x))
    
    def modules(self, 
                   query='', 
                   output_format="DICT(data:list[str])" , 
                   path='./', 
                   n=5, 
                   model='sonnet'):
        front_anchor = f"<{self.anchor}>"
        back_anchor = f"</{self.anchor}>"
        context = c.modules()
        prompt = f"""
        QUERY 
        {query} 
        INSTRUCTION 
        get the top {n} files that match the query
        instead of using the full {os.path.expanduser('~')}, use ~
        CONTEXT
        {context}
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


    def find_files(self, 
                   query='', 
                   output_format="DICT(data:list[str])" , 
                   path='./', 
                   n=5, 
                   model='sonnet'):
        front_anchor = f"<{self.anchor}>"
        back_anchor = f"</{self.anchor}>"
        context = c.files(path)
        prompt = f"""
        QUERY 
        {query} 
        INSTRUCTION 
        get the top {n} files that match the query
        instead of using the full {os.path.expanduser('~')}, use ~
        CONTEXT
        {context}
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
        output = json.loads(output)['data']
        assert len(output) > 0
        return output

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
    
    


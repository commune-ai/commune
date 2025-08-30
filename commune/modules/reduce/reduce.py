
import commune as c
import json
import os


class Reduce:
    description = "This module is used to find files and modules in the current directory"
    agent= c.module('agent')()
    def forward(self,  text, model='google/gemini-2.0-flash-001',  timeout=10, **kwargs):  
        text = f'''
        GOAL
        summarize the following into tupples and make sure you compress as much as oyu can
        CONTEXT
        {text}
        OUTPUT FORMAT IN JSON FORMAT ONLY BETWEEN THE ANCHORS ONLY INCLUDE PURE JSON, NO TEXT
        <OUTPUT>JSON(data:list[])</OUTPUT>
        '''
        assert len(text) < 20000
        return self.process_data(self.ask(text, model=model, stream=0))


    def files(self, path='./',  query='the file that is the core of this folder', timeout=30):

        model = self.model
        future2file = {}
        file2text = {}
        for file in c.files(path):
            try:
                file2text[file] = c.text(file)
                future = c.submit(self.forward, {'text': c.text(file), 'model': model, 'timeout': 10})
            except Exception as e:
                print(f"Error processing {file}: {c.detailed_error(e)}")
                continue
            future2file[future] = file
        files = []
        for future in c.as_completed(future2file, timeout=timeout):
            file = future2file[future]
            try:
                data = future.result()
                if query in data:
                    files.append(data)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        return files
    def process_data(self, data):
        try:
            data = data.split('<OUTPUT>')[1].split('</OUTPUT>')[0]
            data = json.loads(data)
            return data
        except:
            return c

    
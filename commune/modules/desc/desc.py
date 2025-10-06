import commune as c
import os
import json

class Desc:

    def __init__(self, path='~/.desc', key=None):
        self.path = c.abspath(path) 
        self.key = c.key(key)

    def forward(self,   
                module='desc', 
                task= 'summarize the following in the core pieces of info and ignore the rest',
                model='google/gemini-2.0-flash',
                free = True,
                max_age=200,
                cache=True,
                update=False,
                 **kwargs):
                 
        context  = str(c.content(module))
        dirpath = c.dirpath(module)
        cid = c.hash(context)
        path  = c.abspath(f'{self.path}/{module}/{model}.json')
        result = c.get(path, max_age=max_age, update=update)
        if result is not None and cache:
            print(f'Using cached description from {path} (use update=True to refresh)')
            return result['data']
        anchors = ['<JSON_START_DATA>', '</JSON_END_DATA>']
        output_schema = "{'data': str}"
        text = f"""
        --CONTEXT--
        {context}
        --TASK--
        {task}
        --FORMAT--
        write a nice about the module in the following format
        MAKE SURE YOU ONLY OUTPUT WITHIN THE ANCHORS 
        PLEAASE FOLLOW THE FORMAT EXACTLY AND ALWAYS RESPOND IN DICT FORMAT
        OUTPUT_SCHEMA
        DICT(data:str # the description of the module)
        {anchors[0]}JSON_DATA_HERE{anchors[1]}"""

        output = ''
        for ch in c.ask(text, stream=1, model=model, **kwargs):
            output += ch
            print(ch, end='')

        # if not anchors take the first { and last } and parse that
        if anchors[0] not in output or anchors[1] not in output:
            print(f'\n\nNo anchors found, trying to parse first and last curly braces')
            first = output.find('{')
            last = output.rfind('}')
            if first == -1 or last == -1 or first >= last:
                raise ValueError('No valid JSON found in output')
            output = output[first:last+1]
        output =  json.loads(output.split(anchors[0])[-1].split(anchors[1])[0])
        output['context'] = context
        output['cid'] = cid
        output['module'] = module
        output['model'] = model
        output['time'] = c.time()
        output['size'] = len(context)
        output['files'] = [f.replace(dirpath+'/', '') for f in c.files(module)]
        output['signature'] = self.key.sign(output, mode='str')
        signed_data = output.copy()
        print(f'\n\nSigned data: {self.key.verify(signed_data, output["signature"], mode="str")}')
        c.put(path, output)
        return output['data']


    def run(self, mods=None, batch_size=16, timeout=32, trials=3, **kwargs): 
        mods = c.copy(mods or c.core_mods())
        results = {}
        for t in range(trials):
            print(f'Starting trial {t+1}/{trials} with {len(mods)} modules left to describe')
            future2mod = {}
            for mod in mods:
                print(f'Describing {mod}')
                
                
                if len(future2mod) < batch_size:
                    params = {'module': mod, **kwargs}
                    future = (c.submit(self.forward, params, timeout=timeout))
                    future2mod[future] = mod

                else:
                    for f in c.as_completed(future2mod, timeout=timeout):
                        try:
                            result_mod = future2mod.pop(f)
                            results[result_mod] = f.result()
                            futures.remove(f)
                            left_over_mods.remove(result['module'])
                            print(f'Described {result["module"]} with {len(result["data"])} chars')
                        except Exception as e:
                            print(f'Error describing module: {e}')
                    
            mods = [m for m in mods if m not in results]
            print(f'{len(mods)} modules left to describe')
            if len(mods) == 0:
                print('All modules described')
                break
            

        return mods
            



    def mod2desc(self, search=None, **kwargs):
        files = c.files(self.path)
        descs = []
        for file in files:
            desc = c.get(file)
            if desc is None:
                continue
            if search is not None:
                if search.lower() not in desc['data'].lower() and search.lower() not in desc['module'].lower():
                    continue
            descs.append(desc)
        descs = sorted(descs, key=lambda x: x['time'], reverse=True)
        return descs


import commune as c

class DataFolder(c.Module):
    def __init__(self, folder_path: str = './', suffix: str = '.py'):
        config = self.set_config(kwargs=locals())
        self.folder_path = self.resolve_path(config.folder_path)
        self.filepaths = sorted([f for f in self.walk(self.folder_path) if f.endswith('.py')])

    def random_idx(self):
        return self.random_int(0, len(self.filepaths)-1)
    
    def sample(self, idx=None, 
               input_chars:int = 500,
               output_chars: int = 500,
               start_index: int = None,
                real_prob:float=0.5):
        
        if idx == None:
            while True:
                idx = self.random_idx()
                filepath =  self.filepaths[idx]
                file_text = c.get_text(filepath)
                if len(file_text) >= input_chars + output_chars:
                    break
                else:
                    idx = None
        else:
            filepath =  self.filepaths[idx]
            file_text = c.get_text(filepath)

        if start_index == None:
            start_index = c.random_int(0, len(file_text) - input_chars - output_chars )

        #   we need to make sure that the input and output are not the same
        input_bounds = [start_index, start_index + input_chars]
        output_bounds = [start_index + input_chars, start_index + input_chars + output_chars]
        
        sample = {
                'input_text': file_text[input_bounds[0]:input_bounds[1]], 
                'output_text': file_text[output_bounds[0]:output_bounds[1]], 
                'filepath': filepath,
                'idx': idx,
                'start_index': start_index,
                'input_chars': input_chars,
                'output_chars': output_chars,

                 }

        # do a kick flip
        real  = c.random_float(0,1) < real_prob

        sample['real'] = int(real)

        #  then we need to sample a different file
        if sample['real'] == 0 :
            other_sample = self.sample( input_chars=input_chars, real_prob = 1, output_chars=output_chars)
            sample['output_text'] = other_sample['output_text']
    
        return sample


    def test(self, n=100):
        sample = self.sample()
        msg = {'samples_per_second': i / (c.time() - t)}
        c.print(msg)

    @classmethod
    def validate(cls, *objs):
        v = objs[0]




    prompt = '''
    INPUT (JSON):
    ```{sample}```
    QUESTION: 

    WAS THE INPUT REAL (1) OR TAMPERED (0)? -> :

    OUTPUT (answer: int):
    json```
    '''


    def parse_output(self, output:dict)-> dict:
        if '0' in output or 'yes' in output.lower():
            return 0
        elif '1' in output or 'no' in output.lower():
            return 1
        else:
            raise Exception(f'Invalid output: {output}, expected 0 or 1')


    def score(self, model='model', w:float=0.0):

        model = c.connect(model, prefix_match=True)

        try:
            sample = self.sample()
            t = c.time()
            prompt = self.prompt.format(sample=sample)
            output = model.generate(prompt)
            output = self.parse_output(output)
            w = 0.2
        except Exception as e:
            return {'error': c.detailed_error(e), 'w':w}

        if output == sample['real']:
            w = 1

        msg = {
               'prompt': prompt,
               'latency': c.time() - t, 
               'target': sample['real'], 
               'prediction': output,
               'w' : w,
               }

        return msg


    def validate(self, module=None, network=None) -> float:
        if  isinstance(module, str):
            module = c.connect(module, prefix_match=True, network=network)
        if module == None:
            module = self
        t = c.time()
        my_sample = self.sample(real_prob=1)
        kwargs = {k:my_sample[k] for k in ['input_chars', 'output_chars', 'idx', 'start_index']}
        kwargs['real_prob'] = 1
        other_sample = module.sample(**kwargs)


        for k in my_sample.keys():
            if  other_sample[k] != my_sample[k]:
                return 0.0
        return 1.0




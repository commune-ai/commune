import commune as c

class DataTextRealfake(c.Module):

    prompt = '''
    INPUT (JSON):
    ```{sample}```
    QUESTION: 

    WAS THE INPUT REAL (1) OR TAMPERED (0)? -> :

    OUTPUT (answer: int):
    json```
    '''

    def __init__(self, **kwargs):
        config = self.set_config(kwargs=kwargs)
        self.folder_path = self.resolve_path(config.folder_path)
        self.filepaths = sorted([f for f in self.walk(self.folder_path) if f.endswith('.py')])

    def random_idx(self):
        return self.random_int(0, len(self.filepaths)-1)

    
    def sample(self, idx=None, 
               input_chars:int = 500,
               output_chars: int = 500,
               start_index: int = None,
                real_prob:float=0.5, 
                line_break_prob: float = 0.2,
                random_line_ratio: float = 0.2):
        
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
        

        # split the filetext and randomly add line breaks

        sample = {
                'input_text': file_text[input_bounds[0]:input_bounds[1]], 
                'output_text': file_text[output_bounds[0]:output_bounds[1]], 
                'filepath': filepath,
                'idx': idx,
                'start_index': start_index,
                'input_chars': input_chars,
                'output_chars': output_chars,

                 }

        # add line breaks in the input text
        for k in ['input_text', 'output_text']:
            sample[k] = '\n'.join([t + '\n' if c.random_float() > line_break_prob else t for t in sample[k].split('\n')])

        # do a kick flip
        real  = c.random_float(0,1) < real_prob

        sample['real'] = int(real)

        #  then we need to sample a different file
        if sample['real'] == 0 :
            other_sample = self.sample( input_chars=input_chars, real_prob = 1, output_chars=output_chars)
            sample['output_text'] = other_sample['output_text']
    
        return sample


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

import commune as c

class DataTextRealfake(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(kwargs=kwargs)
        self.folder_path = self.resolve_path(config.folder_path)
        self.filepaths = sorted([f for f in self.walk(self.folder_path) if f.endswith('.py')])

    def random_idx(self):
        return self.random_int(0, len(self.filepaths)-1)

    
    def sample(self, idx=None, 
               input_tokens:int = 100,
                output_tokens:int = 100, real_prob:float=0.5):
        if idx is None:
            idx = self.random_idx()
        filepath =  self.filepaths[idx]
        num_lines = len(c.get_text(filepath).split('\n'))
        random_start_line = self.random_int(0, num_lines)

        file_text = c.get_text(filepath)
        file_text = file_text.split('\n')[random_start_line:]
        file_text = '\n'.join(file_text)
        sample = {'input_text': file_text[:input_tokens], 'output_text': file_text[input_tokens:], 'filepath': filepath}

        real  = c.random_float(0,1) > real_prob
        if not real :
            other_sample = self.sample( input_tokens=output_tokens, output_tokens=input_tokens, real_prob = 0)
            sample['output_text'] = other_sample['output_text']
        
        sample['target'] = 1 if real else 0
        return sample


    def test(self, n=100):
        t = c.time()
        for i in range(n):
            sample = self.sample()
            msg = {'samples_per_second': i / (c.time() - t)}
            c.print(msg)
            


    prompt = '''
    INPUT (JSON):
    ```{sample}```
    QUESTION: 

    WAS THE INPUT REAL (1) OR TAMPERED (0)? -> :

    OUTPUT (answer: int):
    json```
    '''


    def parse_output(self, output:dict)-> dict:
        if '0' in output:
            return 0
        elif '1' in output:
            return 1
        else:
            raise Exception(f'Invalid output: {output}, expected 0 or 1')


    def score(self, model='model', w:float=0.0):

        try:
            model_name = model
            model = c.connect(model, prefix_match=True, network='local')
            sample = self.sample()
            t = c.time()
            prompt = self.prompt.format(sample=sample)
            output = model.generate(prompt)
            output = self.parse_output(output)
            w = 0.2
        except Exception as e:
            return {'error': c.detailed_error(e), 'target': sample['target'], 'w':w}

        if output == sample['target']:
            w = 1

        msg = {
               'prompt': prompt,
               'latency': c.time() - t, 
               'target': sample['target'], 
               'prediction': output,
               'w' : w,
               }

        return msg


    def score_models(self, model='model'):

        models = c.servers(model, network='local')
        responses = []
        for model in models:
            msg = self.score(model=model)
            msg['model'] = model
            responses += [msg]

        return responses
            

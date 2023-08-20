import commune as c

class DataTextRealfake(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(kwargs=kwargs)
        self.folder_path = self.resolve_path(config.folder_path)
        self.filepaths = sorted([f for f in self.walk(self.folder_path) if f.endswith('.py')])

    def random_idx(self):
        return self.random_int(0, len(self.filepaths)-1)

    
    def sample(self, idx=None, input_tokens = 100, output_tokens = 100, real_prob:float=0.5):
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
        
        sample['label'] = 1 if real else 0
        return sample


    def test(self, n=100):
        t = c.time()
        for i in range(n):
            sample = self.sample()
        
            msg = {'samples_per_second': i / (c.time() - t)}
            c.print(msg)
            

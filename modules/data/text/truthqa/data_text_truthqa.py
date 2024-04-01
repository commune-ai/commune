import commune as c
from typing import List

class DataTextTruthQa(c.Module):
   
   
    def score_module(self, module:str, fn='sample', kwargs={'idx': 0}):
        reference_output = getattr(self.dataset, fn=fn)(**kwargs)
        if isinstance(module, str):
            output =  c.async_call(module=module,fn=fn, **kwargs)
        else:
            output = getattr(module,fn)(**kwargs)
        if isinstance(output, dict):
            for key in reference_output.keys():
                if key not in output or output[key] != reference_output[key]:
                    w =  0
            w = 1
        else:
            if output == reference_output:
                w = 1
        return {'w': w, 'output': output, 'reference_output': reference_output}

    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.dataset = c.module('data.hf')(**config)
    def sample(self, **kwargs):
        sample =  self.dataset.sample(**kwargs)
        assert isinstance(sample, dict)
        sample = {
            'question': sample['question'],
            'choices': c.shuffle(sample['incorrect_answers'] + sample['correct_answers']),
            'answers': sample['correct_answers']
        }

        sample['answers'] = [i for i, choice in enumerate(sample['choices']) if choice in sample['answers']]
        return sample

    def validate_replicas(self, replicas:List[str], fn='sample', kwargs={'idx': 0}):
        scores = {replica:self.validate(module=replica, fn=fn, kwargs=kwargs) for replica in replicas}
        return scores

    def test(self, n=100):
        t = c.time()
        modules_per_second = 0
        for i in range(n):
            sample = self.sample()
            assert isinstance(sample, dict), f'sample is not a dict: {sample}'
            for key in ['question', 'choices', 'answers']:
                assert key in sample, f'{key} not in sample'
            c.print(i)
            t_passed = c.time() - t
            modules_per_second = i / t_passed
        stats = {
            'modules_per_second': modules_per_second,
            'latency': t_passed, 
            'n': n,
        }

        return {'success': True, 'msg': 'DataTextTruthQa test passed', 'stats': stats }


        
        
        
        



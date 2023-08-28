import commune as c

class DataTextTruthQa(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.dataset = c.module('data.hf')(**config)
    def sample(self, **kwargs):
        return self.dataset.sample(**kwargs)

    def validate(self, module:str , fn='sample', kwargs={'idx': 0}):
        reference_output = getattr(self.dataset, fn=fn)(**kwargs)
        output =  c.async_call(module=module,fn=fn, **kwargs)
        if isinstance(output, dict):
            for key in reference_output.keys():
                if key not in output or output[key] != reference_output[key]:
                    w =  0
            w = 1
        else:
            if output == reference_output:
                w = 1
        return w

    def validate_replicas(self, replicas:List[str], fn='sample', kwargs={'idx': 0}):
        scores = {replica:self.validate(module=replica, fn=fn, kwargs=kwargs) for replica in replicas}
        return scores


        
        
        
        



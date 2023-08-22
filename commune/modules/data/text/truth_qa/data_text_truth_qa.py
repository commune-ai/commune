import commune as c

class DataTextTruthQa(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.dataset = c.module('data.hf')(**config)
    def sample(self, **kwargs):
        return self.dataset.sample(**kwargs)

    def validate(self, fn='sample', kwargs={'idx': 0}):
        reference_output = self.dataset.sample(**kwargs)
        replicas = self.replicas()
        output_map = {}
        jobs = []
        
        outputs = c.gather([c.async_call(replica,fn, **kwargs) for replica in replicas])

        # check if all outputs are the same
        score_map = {}
        for i, output in enumerate(outputs):
            w = 0
            if isinstance(output, dict):
                for key in reference_output.keys():
                    if key not in output or output[key] != reference_output[key]:
                        w =  0
                w = 1
            else:
                if output == reference_output:
                    w = 1
            score_map[replicas[i]] = w
        return score_map



        
        
        
        



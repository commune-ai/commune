import commune as c
Vali = c.module('vali')
class ValiTextMath(Vali):
    def __init__(self, config=None, **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)   
        self.dataset =  c.module(config.dataset)()
        self.init_vali(config)

    def score_module(self, module, **kwargs):
        w = 0

        sample = self.dataset.sample()
        t = c.time()

        prompt = f'''
        INPUT (JSON):
        ```{sample}```
        QUESTION: 

        WAS THE INPUT REAL (1) OR TAMPERED (0)? -> :

        OUTPUT (answer: int):
        '''


        output_text = module.forward(fn='generate', args=[prompt])
        prediction = float(output_text.split("json```")[1].split("```")[0])
        target = float(sample['answer'])
    
        w = abs(prediction - target)

        response = {
            'prompt': prompt,
            'latency': c.time() - t, 
            'target': sample['real'], 
            'prediction': prediction,
            'output_text': output_text,
            'w' : w,
            }

        return response




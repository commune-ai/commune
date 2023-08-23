import commune as c
Vali = c.module('vali')
class ValiTextRealfake(Vali):
    def __init__(self, **kwargs):
        
        config = self.set_config(kwargs=kwargs)   
        Vali.init_vali(self, config=config, **kwargs)
        self.dataset =  c.module(config.dataset)()
        if self.config.start:
            self.start()


    def parse_output(self, output:dict)-> dict:
        if isinstance(output, dict):
            output = c.dict2str(output)

        if '0' in output or 'yes' in output.lower():
            return 0
        elif '1' in output or 'no' in output.lower():
            return 1
        else:
            raise Exception(f'Invalid output: {output}, expected 0 or 1')


    def score_module(self, module):
        w = 0

        try:
            sample = self.dataset.sample()
            t = c.time()
    
            prompt = f'''
            INPUT (JSON):
            ```{sample}```
            QUESTION: 

            WAS THE INPUT REAL (1) OR TAMPERED (0)? -> :

            OUTPUT (answer: int):
            json```
            '''


            output = module.generate(prompt)
            output = self.parse_output(output)

            if output == sample['real']:
                w = 1
            else:
                w = 0.2

            response = {
                'prompt': prompt,
                'latency': c.time() - t, 
                'target': sample['real'], 
                'prediction': output,
                'w' : w,
                }

        except Exception as e:
            response = {'error': c.detailed_error(e), 'w':w}




        return response




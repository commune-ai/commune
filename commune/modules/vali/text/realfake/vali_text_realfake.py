import commune as c
Vali = c.module('vali')
class ValiTextRealfake(Vali):
    def __init__(self, config=None, **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)   
        self.dataset =  c.module(config.dataset)()
        self.init_vali(config)

    def parse_output(self, output:dict)-> dict:
        if isinstance(output, dict):
            output = c.dict2str(output)

        if '0' in output or 'yes' in output.lower():
            return 0
        elif '1' in output or 'no' in output.lower():
            return 1
        else:
            raise Exception(f'Invalid output: {output}, expected 0 or 1')


    def score_module(self, module, **kwargs):
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


            output_text = module.forward(fn='generate', args=[prompt])
            if 'error' in output_text:
                raise Exception(output_text['error'])
            output = self.parse_output(output_text)

            if output == sample['real']:
                w = 1
            else:
                w = 0.2

            response = {
                'prompt': prompt,
                'latency': c.time() - t, 
                'target': sample['real'], 
                'prediction': output,
                'output_text': output_text,
                'w' : w,
                }

        except Exception as e:
            response = {'error': c.detailed_error(e), 'w':0}




        return response




import commune as c
import json
Vali = c.module('vali')

class ValiText(Vali):
    def __init__(self, config = None, **kwargs):
        conifg = self.set_config(config, kwargs=kwargs)
        kwargs['start'] = False
        Vali.__init__(self, config=config, **kwargs)
        self.set_dataset( self.config.dataset )
        self.start()



    def start_dataset(dataset):
        dataset.split('.')[-1]
    def set_dataset(self, dataset:str , **kwargs):
        if c.server_exists(dataset):
            self.dataset = c.connect(dataset)
        else:
            c.module('data.hf').serve(path=dataset.split('.')[-1])
            self.dataset = c.connect(dataset)
        return self.dataset

    def score(self, module='model.openai') -> int:
        if isinstance(module, str):
            module = c.connect(module)
        module.info(timeout=1)
        sample = self.sample()
        prompt = f'COMPLETE THE JSON \n {sample} \n' + " GIVE THE ANSWER AS AN INDEX -> {answer_idx:int} ? \n ```json"
        output = module.generate(prompt, max_tokens=256)
        if isinstance(output, str):
            output = '{' + output.split('{')[-1].split('}')[0] + '}'
            c.print(output)
            output = json.loads(output)
        # if correct answer is in the choices
        if 'answer_idx' not in output:
            answer_idx = int(list(output.values())[0])
        else:
            answer_idx = int(output['answer_idx'])
        if answer_idx in answers:
            w = 1
        else:
            w = 0
        return {'w': w, 'answer_idx': answer_idx, 'answers': answers, 'output': output, 'sample': sample}

    def sample(self):
        # get sample
        sample = self.dataset.sample()
        sample = {
            'question': sample['question'],
            'choices': c.shuffle(sample['incorrect_answers'] + sample['correct_answers']),
            'answers': sample['correct_answers']
        }

        # shuffle choices 
        sample['choices'] = {i: choice for i, choice in enumerate(sample['choices'])}
        sample['answers'] = [i for i, choice in sample['choices'].items() if choice in sample['answers']]

        return sample


    
            


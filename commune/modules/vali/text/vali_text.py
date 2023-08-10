import commune as c
Vali = c.module('vali')

class ValiText(Vali):
    def __init__(self, config = None, **kwargs):
        kwargs['start'] = False
        Vali.__init__(self, config=config, **kwargs)
        self.set_dataset( self.config.dataset )


    def start_dataset(dataset):
        dataset.split('.')[-1]
    def set_dataset(self, dataset:str , **kwargs):
        if c.server_exists(dataset):
            self.dataset = c.connect(dataset)
        else:
            c.module('data.hf').serve(path=dataset.split('.')[-1])
            self.dataset = c.connect(dataset)

        return self.dataset

    def score_module(self, module) -> int:
        # score response
        c.print(module, 'WHADUPFAM')
        sample = self.sample()
        c.print(sample, 'WHADDDDDD')
        return 1

    def sample(self):
        # get sample
        sample = self.dataset.sample()
        sample = {
            'question': sample['question'],
            'choices': sample['incorrect_answers'] + sample['correct_answers'],
            'answer': None,
            'correct_answers': sample['correct_answers']
        }

        # shuffle choices 
        sample['choices'] = c.shuffle(sample['choices'])

        return sample


    
            


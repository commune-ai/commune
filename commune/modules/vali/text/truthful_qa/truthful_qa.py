import commune as c

Vali = c.module('vali')
class ValiTruthfulQA(Vali):

    def __init__(self, config = None, **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        Vali.__init__(self, config=config, **kwargs)
        self.set_dataset(   )
        if c.server_exists(self.config.dataset):
            self.dataset = c.connect(self.config.dataset)
        else:
            c.module('data.hf').serve(path=self.config.dataset.split('.')[-1])

        c.print(self.config)
        # if start:
        #     self.start()

    def start_dataset(dataset):
        dataset.split('.')[-1]


    def score_module(self, module) -> int:
        # score response
        c.print(module, 'WHADUP')
        return 1

    def get_module(self, name:str):
        pass
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


    
            


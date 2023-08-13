import commune as c
Vali = c.module('vali')
class ValiTruthfulQA(Vali):

    def __init__(self, config = None, **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        Vali.__init__(self, config=config, **kwargs)
        self.set_dataset(   )

        c.print(self.config)
        # if start:
        #     self.start()

    def start_dataset(dataset):
        dataset.split('.')[-1]
    def set_dataset(self, dataset = None, **kwargs):
        dataset = dataset or self.config.dataset
        if c.server_exists(dataset):
            self.dataset = c.connect(dataset)
        else:
            c.module('data.hf').serve(path=dataset.split('.')[-1])
            self.dataset = c.connect(dataset)

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


    
            


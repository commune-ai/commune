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

        return self.dataset

    def score_module(self, module='model.vi') -> int:
        module = c.connect(module)
        sample = self.sample()
        for k in ['correct_answers', 'answer']:
            del sample[k]
        sample = c.dict2str(sample)
        prompt = f'COMPLETE THE JSON \n {sample} \n' + " GIVE THE ANSWER AS  {answer_idx:0} ? \n A:"
        output = module.generate(prompt, max_new_tokens=256)
        c.print({'output': output})

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
        sample['choices'] = {i: choice for i, choice in enumerate(sample['choices'])}

        return sample


    
            


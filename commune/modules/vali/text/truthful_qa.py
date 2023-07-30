import commune as c


class ValiTruthfulQA(c.Module):

    def __init__(self, 
                module: str = 'model.text',
                dataset : str = 'data.truthful_qa',
                network : str = 'subspace',
                start: bool = False
    ):
        self.dataset = c.connect(dataset)

        if start:
            self.start()


    def get_module(self, name):

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

    def run_test(self):
        sample = self.sample()
        correct_answers = sample.pop('correct_answers')

        c.print(sample)

    def start(self):
        while True:
            self.run_test()

    
            


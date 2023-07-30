import commune as c


class ValiTruthfulQA(c.Module):

    def __init__(self, 
                module: str = 'model.text',
                dataset : str = 'data.truthful_qa',
                start: bool = False
    ):
        self.dataset = c.connect(dataset)

        if start:
            self.start()

    def run_test(self):
        # get sample
        sample = self.dataset.sample()
        c.print(sample)

    def start(self):
        while True:
            self.run_test()

    
            


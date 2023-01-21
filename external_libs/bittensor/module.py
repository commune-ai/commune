from commune import Module

class ClientModelModule(Module):
    def __init__(self, 
                model=None,
                dataset=None, 
                receptor_pool=None
                **kwargs):
        Module.__init__(self, **kwargs)


        self.model = self.set_model(model)
        self.dataset = self.set_dataset(dataset)

    def set_model(self, model=None):
        if model == None:
            model = self.launch(**self.config['model'])
    
    def set_dataset(self, dataset):
        if dataset == None:
            self.dataset = dataset
        
    

        

    
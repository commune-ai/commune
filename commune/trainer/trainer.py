import tuwang
from typing import List

class Trainer(tuwang.Module):
    def __init__(self,
                experiment:str = 'experiment'):
        self.experiment = experiment
        self.state = {}

        
    def set_experiment(self, experiment:str):
        '''
        Sets the experiment
        '''
        self.experiment = experiment
        return self.experiment
    
    
    def register_trial(self, name:str, stats:dict) -> str:
        '''
        register the stats
        '''
        path = f'{self.experiment}/{name}'
        self.put_json(path, stats)
        
        return path
    
    def list_trials(self, name:str = '') -> List[str]:
        '''
        List the trials
        '''
        return self.glob(self.experiment+'/{name}*')
    
    

if __name__ == "__main__":
    Trainer.run()
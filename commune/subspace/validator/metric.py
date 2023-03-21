

from commune.metric import Metric

import torch

class CrossEntropy(Metric):
    
    def __init__(self,*args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def calculate(self,  value ):
        
        
        input_ids = value.get('input_ids', None)
        pred = value.get('logits', None)
        if input_ids != None:
            gt = input_ids[:, -(pred.shape[1]-1):].flatten()
            pred = pred[:, :-1]
            
        assert isinstance(gt, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
        assert isinstance(pred, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'
        if not hasattr(self, 'loss_fn'): 
            self.loss_fn = torch.nn.CrossEntropyLoss( *self.args, **self.kwargs)
        loss =  self.loss_fn(pred, gt.to(pred.device))
        
        return loss
        
    def set_loss_fn(self, *args, **kwargs) -> None:
        self.loss_fn = torch.nn.CrossEntropyLoss( *self.args, **self.kwargs)
        
    def update( self, value:dict, return_value:bool = False, *params) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        
        loss = self.calculate(value)
        

        self.value = loss.item()
        if return_value:
            return loss.item()
        
        
        return loss
    
    @classmethod
    def test(cls):
        
        print('testing MetricCrossEntropy')
        

    
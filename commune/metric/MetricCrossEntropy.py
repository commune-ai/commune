

from commune.metric import Metric

import torch
class MetricCrossEntropy(Metric):
    

    def calculate(self, gt = None, input = None, pred =None, **kwargs ):
        
        
        if input != None:
            gt = input[:, -(pred.shape[1]-1):].flatten()
            pred = pred[:, :pred.shape[1]-1]
            
        assert isinstance(gt, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
        assert isinstance(pred, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss( *args, **kwargs)
        loss =  loss_fn(pred, gt.to(pred.device))
        
        return loss
        
    
    
    @classmethod
    def update( cls, value:dict, *args, **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        
        value = self.calculate(**value)
        

        self.value = loss.item()
        if return_value:
            return loss.item()
        
        
        return loss
    
    @classmethod
    def test(cls):
        
        print('testing MetricCrossEntropy')
        

    
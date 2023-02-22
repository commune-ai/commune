

from commune.metric import Metric


class CrossEntropy(Metric):
    

    def calculate(pred, gt):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt)
        return loss
    
    
    @classmethod
    def update( cls, pred:torch.Tensor,
                       gt:torch.Tensor = None,
                       input: torch.Tensor=None , 
                       return_value: bool = False,
                       *args, **kwargs) -> torch.Tensor:
        '''
        Calculate the loss for the model.
        '''
        
        
        if input != None:
            gt = input[:, -(pred.shape[1]-1):].flatten()
            pred = pred[:, :pred.shape[1]-1]
            
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        
        assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

        loss_fn = torch.nn.CrossEntropyLoss( *args, **kwargs)
        loss =  loss_fn(pred, gt.to(pred.device))
        
        self.value = loss.item()
        if return_value:
            return loss.item()
        
        
        return loss

    
import commune
import streamlit as st



# model = commune.connect('model::gptj::train2')
model = commune.connect('model:gpt125m')
# model = commune.launch('model.transformer', name='model:gpt125m', kwargs={'model': 'gpt125m'})

# model.train_model(params={'finetune': {'num_layers': 4}}, num_batches=10)
# model = model.launch(name='model:gptj:0', kwargs={'model': 'gptj'})
model.train_model(load=False, save=True, num_batches=10, params={'finetune': {'num_layers': 4}})
# model.save()
# dataset = commune.connect('dataset::bittensor')
# sample = dataset.sample()
# sample['return_keys'] = ['stats']
# sample['train'] = True
# print(model.forward(**sample))
# print(model.model_name)
# model = model('gptj')
# for lr in [0.0001, 0.0002, 0.001]:
#     for i in [1,2,4,8,16]:
#         model.train_model(num_batches= 4000, params= {'finetune': {'num_layers': i}, 'optimizer': {'lr': lr}, 'metrics': {}}, tag= f'base_finetune_{i}_lr_{lr}',save=True, load=True)


# from commune.metric import Metric

# import torch
# class MetricCrossEntropy(Metric):
    
#     def __init__(self,*args, **kwargs):
#         self.args = args
#         self.kwargs = kwargs

#     def calculate(self,  **value ):
        
        
#         input_ids = value.get('input_ids', None)
#         pred = value.get('logits', None)
#         if input_ids != None:
#             gt = input_ids[:, -(pred.shape[1]-1):].flatten()
#             pred = pred[:, :-1]
            
#         assert isinstance(gt, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
#         assert isinstance(pred, torch.Tensor), f'gt is not a torch.Tensor. gt: {gt}'
            
#         if len(pred.shape) == 3:
#             pred = pred.reshape(-1, pred.shape[-1])
        
#         assert gt.shape == pred.shape[:1], f'gt.shape: {gt.shape} pred.shape: {pred.shape}'

#         loss_fn = torch.nn.CrossEntropyLoss( *self.args, **self.kwargs)
#         loss =  loss_fn(pred, gt.to(pred.device))
        
#         return loss
        
    
#     def update( self, value:dict, return_value:bool = False, *args, **kwargs) -> torch.Tensor:
#         '''
#         Calculate the loss for the model.
#         '''
        
#         loss = self.calculate(**value)
        

#         self.value = loss.item()
#         if return_value:
#             return loss.item()
        
        
#         return loss
    
#     @classmethod
#     def test(cls):
        
#         print('testing MetricCrossEntropy')
        


# model = commune.get_module('model.transformer')
# model = model('gpt125m')
# model.train_model()
# tokenizer = model.tokenizer
# dataset = commune.connect('dataset::bittensor')
# model = commune.connect('model::gptneox')
# sample = dataset.sample(no_tokenizer=True)

# sample = tokenizer(sample['text'], max_length=60, truncation=True, padding="max_length", return_tensors="pt")
# t = commune.timer()
# output = model.forward(**sample)
# output['logits'] =(output['logits']).type(torch.float)
# sample.update(output)
# loss = metric.update(sample)
# st.write(loss)
# st.write(t.seconds)

# # print(metrics.set_metric('bob', 5))
# # import torch
# # metrics.rm_metric('bob')
# # adapter_model = commune.get_module('model.adapter')()
# # for i in range(10):
# #     print(adapter_model.train_model(metric_server='metric_server', tag='base1', refresh=True))
# # print(commune.connect('AdapterModel').train_model(num_batches=20, timeout=1))
# # print(MetricMap.test())make d
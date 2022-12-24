import torch
from cortex.model.norm import LayerNorm
from cortex.model.attention import MultiHeadedAttention
import streamlit as st
import copy
import numpy as np
import math
import bittensor
from cortex.metric import  phrase_cross_entropy
from typing import *
class BaseMOE(torch.nn.Module):
    def __init__(self, 
                input_dim:int=10, 
                hidden_dims:List[int]=[128, 64, 32], 
                attention_heads:int=4, 
                output_dim:int=1, 
                metrics:Optional[Dict[str, 'function(pred, gt)']]=None, 
                optimizer:Optional['torch.optim.Optimizer']=None):
                    
        super().__init__()

        self.set_optimizer(optimizer)
        self.set_metrics(metrics)
    

        ## build linear blocks
        self.linear_feedforward = torch.nn.ModuleList([])
        self.linear_norm = torch.nn.ModuleList([])
        self.linear_activation = []
        self.input_norm = LayerNorm(features=input_dim)

        for i,hidden_dim in enumerate(hidden_dims):
            if i == 0:
                linear_input_dim, linear_output_dim = input_dim, hidden_dim
            else:
                linear_input_dim, linear_output_dim = linear_output_dim, hidden_dim

            self.linear_feedforward.append(torch.nn.Linear(linear_input_dim, linear_output_dim))
            self.linear_norm.append(LayerNorm(features=linear_output_dim))
            self.linear_activation.append(torch.nn.ELU())
        
        # Add the multihed attension
        self.endpoint_attention = MultiHeadedAttention(h=attention_heads, d_model=linear_output_dim)

        self.output_feedforward = torch.nn.Linear(linear_output_dim, output_dim )
        self.output_norm = LayerNorm(features=output_dim)


    @property
    def metric2loss_coeff(self):
        coefficients = {
            'phase_cross_entropy': 1
        }


    def calculate_loss_dict(self, pred:torch.Tensor, gt:torch.Tensor) -> Dict[str, Union[int, torch.Tensor]]:
        loss_dict = {}
        total_loss = 0
        metric_input_dict = dict(pred=pred, gt=gt)
        for metric_name, metric_fn in self.metrics.items():
            loss_dict[metric_name] = metric_fn(**metric_input_dict)
            metric2loss_coeff = self.metric2loss_coeff.get(metric_name, 0)
            if metric2loss_coeff>0:
                total_loss += metric2loss_coeff*loss_dict[metric_name]
        loss_dict['loss'] = total_loss
        return loss_dict

    calculate_loss = calculate_loss_dict

    @property
    def default_metrics(self, metrics:dict)-> Dict[str, 'function(prediction,gt)']:
        return dict(
            phase_cross_entropy = lambda prediction, gt: phrase_cross_entropy(target_phrases=gt, topk_tensor=prediction)[0]
        )

    def set_metrics(self, metrics:dict=None):
        self.metrics = {}
        self.metrics = metrics if metrics else self.default_metrics
        self.tracked_metrics = []
        return metrics


    def default_optimizer(self):
        torch.optim.Adam(self.parameters(), lr=0.001)

    @property
    def optimizer(self) -> 'torch.optim.Optimizer':
        if hasattr(self, '_optimizer'):
            self._optimizer = self.default_optimizer

        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer:'torch.optim.Optimizer' ) -> 'torch.optim.Optimizer':
        self._optimizer = optimizer
        return self._optimizer

    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None):
        self.optimizer = optimizer if optimizer else self.optimizer
        return self.optimizer

    def trigger_step(self, loss: torch.Tensor) :       
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, x:dict, )-> dict:
        with torch.no_grad():
            output = self(x)
            return output

    def learn_step(self, x:dict, refresh_tracked_metrics=False):
        if not self.training:
            self.train()
        pred = self(x)
        loss_dict = self.calculate_loss_dict(pred=pred,gt=x['gt'])
        loss = self.trigger_step(loss_dict['loss'])
        return loss.item()

    def forward(self, x):
        tensor_stats_dict = {}
        x_emb = x['endpoint_emb']
        x_pred = x['prediction']
        for i in range(len(self.linear_feedforward)):
            x_emb = self.linear_feedforward[i](x_emb)
            x_emb = self.linear_activation[i](x_emb)
            x_emb = self.linear_norm[i](x_emb)
        routing_scores = torch.nn.functional.softmax( self.output_feedforward(x_emb).squeeze(-1), -1)
        mix_emb = self.mix_response(x_pred, routing_scores)
        return mix_emb

    @staticmethod
    def mix_response( response_success, routing_scores):
        batch_size = response_success[0].shape[0]
        mixed_response = torch.zeros(batch_size, bittensor.__vocab_size__ + 1  , 2)
        all_logits = torch.tensor(list(range(bittensor.__vocab_size__)))
        mixed_response[:, : -1, 1] = all_logits.repeat(batch_size, 1)

        for r, w in list(zip(response_success, routing_scores)):
            response = torch.zeros(batch_size, bittensor.__vocab_size__ , 2)
            response[:, :, 1] = all_logits.repeat(batch_size, 1)

            for batch in range(batch_size):
                r_batch = r[batch, : -1, :]
                r_batch_sorted = r_batch[r_batch[:, 0].sort(descending = False)[1]]
                index = r_batch_sorted[:, 1].long()
                prob = r_batch_sorted[:, 0] 
                response[batch, index, 0] = prob

                mixed_response[batch, :-1, 0] += w[batch] * response[batch, :, 0].clone()

        for batch in range(batch_size):
            mixed_response[batch, -1, :] = torch.tensor([[0, -1]])

        return mixed_response


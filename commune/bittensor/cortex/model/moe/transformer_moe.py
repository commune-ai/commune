import torch
from cortex.model.norm import LayerNorm
from cortex.model.attention import MultiHeadedAttention
import streamlit as st
import copy
import numpy as np
import math


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def tensor_stats(x):
    return {'std': x.std().item(), 'mean': x.mean().item()}

class BaseMOE(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[128, 64, 32] , attention_heads=4, output_dim=1000, metrics=None, optimizer=None):
        
        super().__init__()
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
            linear_feedforward = torch.nn.Linear(linear_input_dim, linear_output_dim)
            self.linear_feedforward.append(linear_feedforward)
            linear_norm = LayerNorm(features=linear_output_dim)
            self.linear_norm.append(linear_norm)
            self.linear_activation.append(torch.nn.ELU())
        
        self.endpoint_attention = MultiHeadedAttention(heads=attention_heads, d_model=linear_output_dim)
        self.sequence_attention = MultiHeadedAttention(heads=attention_heads, d_model=linear_output_dim)

        self.output_feedforward = torch.nn.Linear(linear_output_dim, output_dim )
        self.output_norm = LayerNorm(features=output_dim)
        self.set_optimizer(optimizer)
        self.set_metrics(metrics)
    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def calculate_metrics(self, pediction, gt):
        metrics = {}
        metrics['cross_entropy'] =  self.metrics['cross_entropy'](pediction, gt)
        return metrics

    def calculate_loss(self, pediction, gt):
        loss =  self.metrics['cross_entropy'](pediction, gt)
        return loss

    def set_metrics(self, metrics=None):
        self.metrics = {}
        if metrics == None:
            self.metrics['cross_entropy'] =  torch.nn.CrossEntropyLoss()
        return metrics
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        if optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return self.optimizer


    def learn_step(self, input, gt):
        pred = self(input)
        loss = self.calculate_loss(pediction=pred.reshape(-1, pred.shape[-1]),gt=gt.flatten())  
        
        self.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, x):
        tensor_stats_dict = {}
        x = self.input_norm(x)
        tensor_stats_dict[f'input']= tensor_stats(x)
        for i in range(len(self.linear_feedforward)):
            
            x = self.linear_feedforward[i](x)
            x = self.linear_activation[i](x)
            x = self.linear_norm[i](x)

        
        batch_size, num_endpoints, seqeunce_length, vector_dim = x.shape
        # x = x.transpose(1,2).reshape(-1, num_endpoints, vector_dim )
        # x = self.endpoint_attention(key=x, value=x, query=x)
        # x = x.reshape(batch_size, num_endpoints, seqeunce_length, vector_dim )
        x = x.mean(dim=1)

        seqeunce_attn_mask = self.subsequent_mask(seqeunce_length).to(x.device)
        # x = self.sequence_attention(key=x, value=x, query=x, mask=seqeunce_attn_mask)
        x = self.output_feedforward(x)

        x = torch.nn.functional.softmax(x, -1)
        st.write(x.shape)

        # st.write(tensor_stats(x),  'output')
        return x

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
            
            mixed_response[:, :-1, 0] += w * response[:, :, 0]

        for batch in range(batch_size):
            mixed_response[batch, -1, :] = torch.tensor([[0, -1]])

        return mixed_response


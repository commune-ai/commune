import os, sys
from pprint import pp

from functools import partial
import asyncio
from copy import deepcopy
from typing import Union, Optional
from concurrent import futures
import os, sys
from typing import *
from loguru import logger
import time
from munch import Munch
import argparse
import torch
from commune.utils.torch import tensor_dict_info
from commune.utils.tokenizer import decode_topk
import streamlit as st
# logger = logger.opt(colors=True)
import commune
commune.new_event_loop()
if os.getenv('USE_STREAMLIT') == 'true':
    import streamlit as st
import os
    
# import torch
import commune
# commune.utils
from torch import nn
from torch import Tensor
"""
Examples 



"""


class MultiheadAttention(torch.nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights



class LayerBlock(torch.nn.Module):

    def __init__(self, in_dim:int=10, out_dim:int=10, norm_fn:Callable = 'layer', act_fn:str = 'gelu'):
        super(LayerBlock, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # self.W = torch.nn.Parameter(torch.randn(self.in_dim, self.out_dim))
        # self.b = torch.nn.Parameter(torch.randn(self.out_dim))
        self.layer = torch.nn.Linear(self.in_dim, self.out_dim)
        self.norm_fn = self.set_norm_fn(norm_fn)
    
        self.act_fn = self.set_act_fn(act_fn)

    norm_fn_map = {
        'layer': 'LayerNorm',
        'group': 'GroupNorm',
        'batch': 'BatchNorm',
    }
    def set_norm_fn(self, norm_fn:str, **kwargs):
        if norm_fn == None:
            norm_fn = lambda x: x
        elif isinstance(norm_fn, str):
            norm_fn = self.norm_fn_map.get(norm_fn, norm_fn)
            if norm_fn == 'LayerNorm':
                kwargs = {'normalized_shape': self.out_dim}
            norm_fn = getattr(torch.nn, norm_fn)(**kwargs)
        self.norm_fn = norm_fn
        
        return self.norm_fn
    act_fn_map = {
        'relu': 'ReLU',
        'gelu': 'GELU',
        'tanh': 'Tanh',
        'sigmoid': 'Sigmoid',
        'softmax': 'Softmax'
    }
    def set_act_fn(self, act_fn:str):
        if isinstance(act_fn, str):
            act_fn = self.act_fn_map.get(act_fn, act_fn)
            act_fn = getattr(torch.nn, act_fn)()
        elif act_fn == None :
            act_fn = lambda x: x   
        else:
            raise ValueError(f'Activation function {act_fn} not found')   
        
        self.act_fn = act_fn
        return self.act_fn
        # initialize the parameters
    def init_weights(self):
        in_d = self.W.shape[0]
        y = 1.0/np.sqrt(in_d)
        self.W.data.uniform_(-y, y)
        self.b.data.fill_(0)

    def forward(self, x:torch.Tensor, choice = 'left'):
        
        x = x[..., :self.in_dim].to(self.layer.weight.device)
        # cast x to the same device as the layer weights
        x = x.to(self.layer.weight.dtype) # cast to the same dtype as the weights
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        
        emb = self.layer(x)
        # emb = torch.matmul(x.half(), self.W) + self.b
        # emb = torch.einsum('ij,bi -> bj', [self.W, x]) + self.b
        emb = self.act_fn(emb)     
        emb = self.norm_fn(emb)

        emb = emb.reshape(*original_shape[:-1], emb.shape[-1])
        
        return emb
    



class AdapterModel(torch.nn.Module, commune.Module):
    def __init__(self, 
                 in_dim = 10,
                 hidden_dim:int=64,
                 num_layers:int=8,
                 device: str = 'cpu',
                 out_dim: Optional[int] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        torch.nn.Module.__init__(self)
        self.build(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, device=device)
        self.set_optimizer(**(optimizer if optimizer != None else {}))
        self.set_device(device)
        
    @property
    def device(self) -> str:
        return self._device
    
    def set_device(self, device:str) -> str:
        self.to(device)
        self._device = device
        return device
    def build(self, in_dim:int,
              hidden_dim:int, 
              out_dim:int, 
              device='cpu',
              num_layers:int=1):
        if out_dim == None:
            out_dim = in_dim
        
        # build the encoder
        encoder_blocks = [LayerBlock(in_dim, hidden_dim)]
        for i in range(num_layers):
            encoder_blocks.append(LayerBlock(hidden_dim, hidden_dim, norm_fn='layer', act_fn='gelu'))
        self.encoder = torch.nn.Sequential(*encoder_blocks)
        
        # build the decoder
        
        decoder_blocks = []
        for i in range(num_layers):
            decoder_blocks.append(LayerBlock(hidden_dim, hidden_dim, norm_fn='layer', act_fn='gelu'))
        
        decoder_blocks += [LayerBlock(hidden_dim, out_dim, norm_fn=None, act_fn=None)]
        self.decoder = torch.nn.Sequential(*decoder_blocks)
    def forward(self, x):
        emb = self.encoder(x.to(self.device))
        emb = self.decoder(emb)
        emb = torch.nn.Softmax(dim=-1)(emb)
        emb = torch.log(emb + 1e-8)
        return emb

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def loss(self, x):
        return torch.nn.functional.mse_loss(self.forward(x), x.to(self.device))

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f'AdapterModel()'

    def __str__(self):
        return f'AdapterModel()'
    

    
    def set_optimizer(self, **params) -> 'optimizer':
        self.optimizer = self.get_optimizer(**params)
        return self.optimizer
    
    def get_optimizer(self, optimizer=None, **params) -> 'optimizer':
        if optimizer == None:
            optimizer =  torch.optim.Adam
        elif isinstance(optimizer, str):
            optimizer = commune.import_object(optimizer_class)
        elif isinstance(optimizer, type):
            return optimizer_class
        
        
        params = params.pop('params', {'lr': 0.001})
        optimizer = optimizer(self.parameters(), **params)
        
        return optimizer
    


class EnsembleModel( nn.Module, commune.Module):

    def __init__(self,
                models: List[str] = ['model::gpt125m', 'model::gptj', 'model::gpt3b'],
                # models: List[str] = ['model::gpt125m'],
                hidden_dim = 256,
                tokenizer: 'tokenizer' = 'bittensor',
                optimizer:  'torch.optimizer' = None,
                metrics: Dict= None,
                load: bool = True,
                device: str = 'cuda',
                tag: str = None,
                ):
        nn.Module.__init__(self)
        self.tag = tag
        
        self.model_device = 'cpu'
        
        self.model_name = 'ensemble'
        

        # set model and tokenizer
        self.set_model(models=models, hidden_dim=hidden_dim, device=device)

        self.set_tokenizer(tokenizer=tokenizer if tokenizer != None else self.model_name)


        # set tokenizer to model name (HF only) if tokenizer == None
        
        self.set_optimizer(optimizer=optimizer)
        
        self.set_metrics(metrics=metrics)
        
        self.set_stats()
        
        if load:
            self.load()
        
        
        
    def set_optimizer(self, optimizer:'torch.optim.Optimizer'=None, *args, **kwargs):
        
        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        self.optimizer = optimizer
        return self.optimizer
    @classmethod
    def test(cls, topk=1024, output_length=10):
        
        model = cls()
        dataset = commune.connect('dataset::bittensor')
        model = model.to('cuda')
        for i in range(100):
            sample = dataset.sample(sequence_length=256)
            loss = model.learn_step(**sample)
            
        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
     

    @classmethod
    def calculate_loss( cls, pred, gt = None, input=None , *args, **kwargs):
        if input != None:

            gt = input[:, -pred.shape[1]:].flatten()
        if len(pred.shape) == 3:
            pred = pred.reshape(-1, pred.shape[-1])
        loss_fn = torch.nn.CrossEntropyLoss( *args, **kwargs)
        loss =  loss_fn(pred, gt.to(pred.device))
        return loss

    def set_metrics(self, metrics=None):
        self.metrics = {}
        if metrics == None:
            self.metrics['cross_entropy'] =  torch.nn.CrossEntropyLoss()
        return metrics
    
    async def async_model_forward(self, model, *args, **kwargs):
        return self.models[model].forward(*args, **kwargs)
        
        
        
    def aggregate(self, 
                  x: List[torch.Tensor], 
                  *args, **kwargs) -> Dict[str, torch.Tensor]:
        
        
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = torch.sum(x, dim=0)
        x = torch.softmax(x, dim=-1)
        x = torch.log(x + 1e-10)
        
        return x
    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def forward(self, *args, output_length=10, topk=512, return_topk_only=True,  **kwargs):
        kwargs.update(dict(
            output_hidden_states=True,
            hidden_dim_bounds = None,
            output_logits=False, 
            output_topk=True, 
            output_length=output_length,
            token_remap = False , 
            logit_remap = False,
            topk=topk
        ))
        
        kwargs['output_hidden_states'] = True
        jobs = []
        
        for model in self.models:
            job = self.async_model_forward(model=model, *args, **kwargs)
            jobs.append(job)
            
        # return 
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        peer_outputs =  loop.run_until_complete(asyncio.gather(*jobs))
        
        max_token_index = max([t['topk'][:,:, kwargs['topk']:].max().item() for t in peer_outputs])
        
        model_names = self.model_names
        for model_i, peer_output in enumerate(peer_outputs):
            if 'hidden_states' in peer_output:
                peer_output['hidden_states'] = peer_output['hidden_states'][..., :self.hidden_dim]
                model_name = model_names[model_i]
                adapter_emb = self.model_adapter[model_name](peer_output['hidden_states'])
                peer_output['routing_score'] = self.routing_layer(adapter_emb)
            if 'topk'  in peer_output:
                peer_output['logits'] = decode_topk(peer_output['topk'], vocab_size=int(max_token_index+1), topk= kwargs['topk'])
                peer_outputs[model_i] = peer_output
            else:
                peer_outputs[model_i] = peer_output
            
        # stack with different logits dimensions and aggregate
        
        routing_scores = torch.cat([x['routing_score'] for x in peer_outputs], dim=-1)
        
        routing_scores = torch.softmax(routing_scores, dim=-1).to(self.device)
        
        output_dict = dict(
            # peer_logits = torch.stack([x['logits'] for x in peer_outputs], dim=0),
            peer_hidden_states = torch.stack([x['hidden_states'] for x in peer_outputs], dim=0),
            peer_logits = [],
            peer_loss_per_token = [],
            peer_loss_per_sentence = [],
            peer_loss = []
            
        )
        
        # calculate score per token and sentence
        for model_i, peer_output in enumerate(peer_outputs):
            pred =  peer_output['logits'][:,:-1]
            peer_loss = self.calculate_loss(pred=pred, input=kwargs['input_ids'], reduction='none')
            peer_loss = peer_loss.reshape(pred.shape[0], -1)
            output_dict['peer_loss_per_token'].append(peer_loss)
            output_dict['peer_loss_per_sentence'].append(peer_loss.mean(dim=-1))
            output_dict['peer_loss'].append(peer_loss.mean())
            
            
            
        output_dict['peer_loss_per_sentence'] = torch.stack(output_dict['peer_loss_per_sentence'], dim=0)
        output_dict['peer_loss_per_token'] = torch.stack(output_dict['peer_loss_per_token'], dim=0)
        output_dict['peer_loss'] = torch.stack(output_dict['peer_loss'], dim=0)
        output_dict['best_peer_id'] = torch.argmin(output_dict['peer_loss'], dim=0)
        output_dict['best_peer_loss'] = torch.index_select(output_dict['peer_loss'],  index = output_dict['best_peer_id'] , dim=0)
        prior_routing_scores = output_dict['peer_loss_per_sentence'] 
        prior_routing_scores =  (prior_routing_scores - prior_routing_scores.max(0).values[None, ...])/(prior_routing_scores.std(0)[None,...] + 1E-10)
        
        prior_routing_scores = prior_routing_scores / prior_routing_scores.sum(0)[None, :]
        
        
        
        
        
        # calculate the routing scores
        for model_i, peer_output in enumerate(peer_outputs):
            pred = peer_output['logits']
            peer_score = prior_routing_scores[model_i, ...][:]
            
            output_dict['peer_logits'] += [torch.einsum('ijk,i -> ijk', pred , prior_routing_scores[model_i])]
        
        output_dict['logits'] = (self.aggregate(output_dict['peer_logits']).to('cuda') )
        
        pred = output_dict['logits']
        
        
        output_dict['ensemble_loss_per_token'] = self.calculate_loss(pred=output_dict['logits'][:,:-1], input=kwargs['input_ids'], reduction='none')
        output_dict['ensemble_loss_per_token'] = output_dict['ensemble_loss_per_token'].reshape(-1, pred.shape[1]-1)
        output_dict['ensemble_loss_per_sentence'] = output_dict['ensemble_loss_per_token'].mean(-1)
        
        
        
        
        output_dict['ensemble_loss'] = output_dict['ensemble_loss_per_sentence'].mean()
        
        
        st.write ('Stats Fam')
        st.write('Ensemble Loss', output_dict['ensemble_loss'])
        st.write('Peer Loss Per Sentence', output_dict['peer_loss'])
        st.write('Best Peer Loss', output_dict['best_peer_loss'])

        
        return Munch(output_dict)
    

    @property
    def device(self):
        # deepspeed has .module.device to access device
        return self.model_device

    def set_model(self, models:List[str], hidden_dim:int=256, load:bool=True, device:str = None):
        
        self.model_adapter = nn.ModuleDict()
        
        self.models = {} 
        for model in models:
            self.models[model] = commune.connect(model)
            self.model_adapter[model] = AdapterModel(in_dim=hidden_dim, out_dim=128, hidden_dim=256)
        
        self.config = Munch(self.models[model].model_config)
        self.hidden_dim = hidden_dim
        self.routing_layer = LayerBlock(in_dim=128, out_dim=1, norm_fn=None, act_fn=None)
        return self.models
    

    def list_models(self):
        return list(self.models.keys())

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    shortcuts = {
        'gptj': 'EleutherAI/gpt-j-6B',
    }
    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None]):
        from transformers import AutoTokenizer
        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
        if isinstance(tokenizer, str):
            if tokenizer == 'bittensor':
                import bittensor
                tokenizer = bittensor.tokenizer()
            else:
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except ValueError:
                    print('resorting ot use_fast = False')
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.tokenizer = tokenizer
        
        

        if  self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        return self.tokenizer

    @staticmethod
    def encode_topk( forward_response_tensor: torch.Tensor , topk:int=4096) -> torch.Tensor:
        """ Returns topk tokens/probabilities given unnormalized logits as input. """

        #import ipdb; ipdb.set_trace()

        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

        topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
        # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

        topk_values = probs.gather( index=topk_indices, dim=-1)
        encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_probs  # [batch_size, sequence_len, topk + topk]

    def tokenize(self, text: str = 'Whadup', input_ids_only:bool = True, device: str=None) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        device = device if device != None else self.device
        tokenizer_output = self.tokenizer(text, return_tensors='pt')
        if input_ids_only:
            return tokenizer_output.input_ids.to(self.device)
        return self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)

    
    def learn_step(self, **sample ):
        targets = sample['input_ids'][:,1:].to(self.device)
        sample['input_ids'] = sample['input_ids'][:,:-1].to(self.device)
        self.optimizer.zero_grad()
        
        
        with torch.autocast(device_type='cuda'):
            pred = self.forward(**sample, no_grad=False)
            logits =  pred['logits']
            targets = targets[:,-logits.shape[1]:]
            pred = logits.reshape(-1, logits.size(-1))
            loss = self.calculate_loss(pred=logits.reshape(-1, logits.size(-1)).to(self.device), 
                                        gt=targets.flatten().to(self.device))              
        

        loss.backward()
        self.optimizer.step()
    
        
        return loss.item()

    def set_stats(self, stats:dict=None): 
        if stats == None:
            stats =  dict(
                steps = 0,
                loss = 0,
            )
        self.stats = Munch(stats)
        

    @property
    def module_tag(self): 
        return self.resolve_module_tag()
    
    def resolve_module_tag(self, tag=None):
        tag = tag if tag else self.tag
        module_tag = self.model_name.replace("/", "_")
        if tag:
            module_tag +=  f'_{tag}'
        return module_tag
    
    def save(self, tag:str = None, trainable_only:bool = True):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        model_state_dict = self.models.state_dict()
        
        if trainable_only:
            model_state_dict = {k:v for k,v in model_state_dict.items() if v.requires_grad} 
    
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'stats': dict(self.stats)
        }
    
        torch.save(state_dict, path)
        
        return path
    
    def load(self, tag=None):
        module_tag = self.resolve_module_tag(tag=tag)
        path = self.resolve_path(module_tag)
        if not os.path.exists(path):
            logger.warning(f'No saved model found at {path}')
            return
        loaded_state  = torch.load( path)
        state_dict = self.models.state_dict()
        for k,v in loaded_state['model'].items():
            assert k in state_dict
            state_dict[k] = v
        self.models.load_state_dict(state_dict)
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.set_stats(loaded_state['stats'])
        
    @classmethod
    def local_train(cls, 
                    tag:str = 'demo', 
                    num_batches:int = 200,
                    num_epochs:int = 200, 
                    dataset:str= 'BittensorDataset', **kwargs):
        model = cls(tag=tag, load=True,  **kwargs)
        dataset = cls.connect(dataset)
        
        best_loss = 10e10
        for epoch in range(num_epochs):
            total_epoch_loss = 0
            epoch_loss = 0
            if epoch > 0:
                model.load(tag=tag)
            for i in range(num_batches):
                sample = dataset.sample()
                loss = model.learn_step(**sample)
                try:
                    total_epoch_loss += loss
                except:
                    continue
                epoch_loss = total_epoch_loss/(i+1)
                info_str = f'Batch {i}/{num_batches} Epoch {epoch}/{num_epochs} CE: {loss} Epoch Loss: {epoch_loss} Best Loss: {best_loss}'
                logger.success(info_str)
                print('BROOO')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                try:
                    model.save(tag=tag)
                except TypeError:
                    continue

  
        
    def loss(self, logits: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:

        if not hasattr(self, 'loss_fct'):
            self.loss_fn = torch.nn.CrossEntropyLoss()
            
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return loss

    @classmethod
    def test_neuron(cls, tokenizer='bittensor', num_batches=10, dataset='dataset::bittensor', batch_size=32, sequence_length=12, topk=4096, **model_kwargs):
        from commune.block.bittensor.neuron.miner import neuron
        from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_token_phrases, prep_tokenizer
        self = cls( tokenizer=tokenizer)
        self.to('cuda')
        nucleus = neuron(model=self).model
        nucleus.model.train()
        nucleus.model.eval()
        nucleus.model = nucleus.model.half()
        nucleus.model.config.hidden_size
        nucleus.model.config.pad_token_id
        nucleus.model.config.eos_token_id
        nucleus.model.named_parameters()
        state_dict = nucleus.model.state_dict()
        nucleus.model.load_state_dict(state_dict)
        
        dataset = commune.connect(dataset)
        sample = dataset.sample()
        
        for i in range(num_batches):
            sample = dataset.sample(batch_size=32, sequence_length=256)
            target = sample['input_ids'][:, -1:] 
            inputs_x = sample['input_ids'][:, :-1] 
            t = commune.timer()
            message, _model_output, topk_tensor = nucleus.encode_forward_causallmnext(inputs_x, topk=topk)
            loss_tuple = phrase_cross_entropy(topk_tensor=topk_tensor, target_phrases=target)
            commune.print(f'Loss : {loss_tuple[0].item()} Time: {t.seconds}', 'cyan')
 
    @classmethod
    def run_neuron(cls, tokenizer='bittensor'):
        import bittensor
        from commune.block.bittensor.neuron.miner import neuron
        self = cls( tokenizer=tokenizer)
        n = neuron(model=self)  
        n.run()
 
if __name__ == "__main__":
    
    
    EnsembleModel.run_neuron()
    # EnsembleModel.test_neuron()
    # print('FUCK')
    # TransformerModel('gptj', tag='demo', load=True).save_pretrained()
    
    # TransformerModel.run()
    # TransformerModel.experiment()



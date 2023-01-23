#!/bin/python3
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
""" The bittensor base validator

Example:
    $ python3 miners/text/core_validator.py --logging.debug

"""
import argparse
import time
import datetime
import bittensor
import torch
import os
import wandb
import math
import random
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.errors import MarkupError
from rich.traceback import install
from typing import List, Tuple, Callable, Dict, Any, Union, Set

from bittensor._neuron.text.neuron_utilities import ThreadQueue, PositionalEncoding, calc_loss_fct
from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_tokens_to_vocab_size, prune_tokens

from torch.nn.functional import kl_div
from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
from threading import Lock
from prometheus_client import Counter, Gauge, Histogram, Summary, Info

from utils import *
logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

# Neuron stats recorded by validator neuron/nucleus
#   [Column_name, key_name, format_string, rich_style]  # description


logger = logger.opt( colors=True )

class nucleus( torch.nn.Module ):
    """ Nucleus class which holds the validator model.
    """
    def __init__( self, config, device, subtensor ):
        super(nucleus, self).__init__()
        self.config = config

        self.config.nucleus.scaling_law_power = subtensor.scaling_law_power if self.config.nucleus.scaling_law_power == -1 else self.config.nucleus.scaling_law_power
        self.config.nucleus.synergy_scaling_law_power = subtensor.synergy_scaling_law_power if self.config.nucleus.synergy_scaling_law_power == -1 else self.config.nucleus.synergy_scaling_law_power

        self.device = device
        self.max_n = subtensor.max_n
        self.permute_uids = []  # iterable of next UIDs to query, reset to permuted UIDs when empty

        tokenizer = bittensor.tokenizer()
        self.pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]

        # Token embeddings project int64 tokens onto representations.
        self.token_embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        
        # Routing encoder, projects token embeddings onto context for routing inputs.
        self.routing_encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.routing_encoder = TransformerEncoder( self.routing_encoder_layers, 1 )

        # Encoder projects response representations onto hidden units.
        self.encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.encoder = TransformerEncoder( self.encoder_layers, config.nucleus.nlayers )

        # Decoder which projects hidden unit representations on to the token dimension.
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Positional Encoding
        self.local_pos_encoder = PositionalEncoding( bittensor.__network_dim__, self.config.nucleus.dropout )

        # Crosss entropy loss for NTP.    
        self.loss_fct = torch.nn.CrossEntropyLoss()
    
        # SGMOE Gates: Instantiating the gates per expert.
        self.gates = torch.nn.Linear( bittensor.__network_dim__, self.max_n, bias=True ).to( self.device )

        self.sigmoid = torch.nn.Sigmoid()

        self.reset_weights()

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default = 20 )
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200 )
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default = 2 )
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2 )
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.importance', type=float, help='hyperparameter for the importance loss', default=3)
        parser.add_argument('--nucleus.noise_multiplier', type=float, help='Standard deviation multipler on weights', default=2 )
        parser.add_argument('--nucleus.no_dendrite_backward', action='store_true', help='Pass backward request to the server side or not', default=False )
        parser.add_argument('--nucleus.scaling_law_power', type=float, help='Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5. (default value: -1, pulling from subtensor directly)', default=-1)
        parser.add_argument('--nucleus.synergy_scaling_law_power', type=float, help='Power for synergy modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5. (default value: -1, pulling from subtensor directly)', default=-1)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def reset_weights ( self ):
        r""" Resets the validator weights.
        """
        # === Resets all the weights using xavier initialization. ===
        torch.nn.init.xavier_uniform_ ( self.token_embedding.weight )
        torch.nn.init.xavier_uniform_ ( self.decoder.weight )
        torch.nn.init.xavier_uniform_( self.gates.weight )
        def init_xavier( component ):
            try:
                torch.nn.init.xavier_uniform_( component.weight )
            except: pass
        self.routing_encoder.apply( init_xavier )
        self.encoder.apply( init_xavier )
        torch.nn.init.xavier_uniform_( self.gates.weight )
    
    def forward(
            self,
            inputs: torch.FloatTensor,
            metagraph: 'bittensor.Metagraph',
            dendrite: 'bittensor.Dendrite',
    ):
        r"""
        Forward validator pass. Selects endpoints to query and validate, calculates routing_score and Shapley values
        for validated synapses.
            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    Tensor inputs to distribute to neurons using query context.
                metagraph (bittensor.Metagraph):
                    Metagraph object used to query network information.
                dendrite (bittensor.Dendrite):
                    Dendrite RPC client used to make network queries.
            Returns:
                loss (:obj:`torch.FloatTensor`):
                    Loss for training validator nucleus and dendrite backward to endpoints.
                neuron_stats (:obj:`Dict`, `required`):
                    Statistics per endpoint for this batch.
        """
        start_time = time.time()

        val_len = self.config.neuron.validation_len  # Number of tokens to holdout for phrase validation beyond sequence context
        prune_len = self.config.neuron.prune_len  # Number of tokens to prune from each validation input sequence
        inputs = prune_tokens(inputs.to(self.device), prune_len=prune_len, margin=val_len+3)  # prune input sequence without last validation tokens [batch_size, sequence_len]
        inputs_seq = inputs[..., :-val_len]  # sequence without validation tokens [batch_size, sequence_len]

        # === Create the local context used to select endpoints ===
        # The context tensor returns a hidden unit representation for the text inputs
        # this context can be used as input to the gates in the next step.
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        embedding = self.token_embedding(inputs_seq) * math.sqrt(bittensor.__network_dim__)

        # === Create an attention mask ===
        # The attention mask will mask out parts of the context
        # This prevents cheating and forward-looking when predicting each token in the sequence.
        # src_mask: (torch.FloatTensor) attention mask adds -inf to positions not allowed to attend
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.device)

        # === Apply the positional encoding to help select endpoints ===
        # The positional encoder provides information based on the relative postion of each token
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        # pos_embedding: (torch.FloatTensor) positional encoded embedding.
        # pos_embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        pos_embedding = self.local_pos_encoder(embedding)

        # routing_context: (torch.FloatTensor): context tensor which is used to select endpoints.
        # routing_context.shape = [ batch size, __network_dim__ ]
        routing_context = self.routing_encoder(pos_embedding, mask=src_mask)

        # === Get gate values for UIDs. ===
        # We iterate over each of the network UIDs and compute a querying score for each
        # using the gating function. This returns a score per endpoint per example.
        # routing_score: (torch.FloatTensor): score per example, per endpoint.
        # routing_score.shape = [metagraph.n]
        # The gates act over the last embedding of the routing_context.
        routing_score = torch.mean(self.sigmoid(self.gates(routing_context[:, -1, :])), dim=0)

        # Ensure number of queried neurons does not exceed metagraph.n
        num_endpoints = min([self.config.nucleus.topk, metagraph.n])

        # === Ensure each UID is queried once ===
        # Persist object variable self.permute_uids across forward calls.
        # Reset to new permutation of all UIDs once empty.
        if len(self.permute_uids) == 0:  # no more UIDs to query
            self.permute_uids = torch.randperm(metagraph.n)  # reset to new permutation of all UIDs

        # === Randomly select num_endpoints UIDs ===
        random_uids = self.permute_uids[:num_endpoints]  # newest selection of UIDs to query
        self.permute_uids = self.permute_uids[num_endpoints:]  # slice out remaining selection

        # === Get endpoint information for the selected UIDs ===
        # We index into the metagraph's endpoints and return a list of the filtered set of endpoints we wish to query.
        # random_endpoints: List[bittensor.endpoints]: endpoint information for filtered uids.
        # len(neurons) == self.config.nucleus.topk
        random_endpoints = [metagraph.endpoints[uid] for uid in random_uids]
        num_endpoints = len(random_endpoints)  # in case len(self.permute_uids) < num_endpoints during random_uids select

        logger.info(f'Forward \t| Routing forward <dim>[{time.time() - start_time:.3g}s]</dim>')
        logger.info(f'Dendrite \t| Request {num_endpoints} x {list(inputs_seq.shape)}')
        request_start_time = time.time()

        # === Define which synapse we want to use ===
        # The synapse defines the task we are sending to the neurons
        # synapses: List[bittensor.synapse]: synapse information
        # TODO: WORK IN PROGRESS, prototype
        if self.config.neuron.validation_synapse == 'TextCausalLMNext':
            synapses = [(bittensor.synapse.TextCausalLMNext(), textcausallmnext)]
        else: 
            synapses = [(bittensor.synapse.TextCausalLM(), textcausallm)]

        # === Query the endpoints ===
        # Makes the dendrite call into the network returning the representations
        # for each of the endpoints. The return ops can be used to filter weights and outputs.
        # query_responses: (List[torch.float64]): responses from each endpoint.
        # query_responses.shape = self.config.nucleus.topk * num_synapses * [batch_size, sequence_len, synapse_dim]
        # return_ops: (torch.int64): Return ops.
        # return_ops.shape = self.config.nucleus.topk * [num_synapses]
        query_responses, return_ops, times = dendrite.text(
            endpoints=random_endpoints,
            inputs=inputs_seq,
            synapses=[syn for syn, _ in synapses],
            timeout=bittensor.__blocktime__
        )

        if self.config.nucleus.no_dendrite_backward:
            query_responses = [[syn.detach().to(self.device) for syn in res] for res in query_responses]
            return_ops = [ops.detach().to(self.device) for ops in return_ops]
            times = [t.detach().to(self.device) for t in times]

        # Send responses to device. This is required to ensure we move the responses
        # Onto the correct device.
        for responses in query_responses:
            for response in responses:
                response.to(self.device)

        logger.info(f'Dendrite \t| Request {num_endpoints} x {list(inputs_seq.shape)} '
                    f'<dim>[{time.time() - request_start_time:.3g}s]</dim>')

        # === Prepare validation parameter set ===
        console_width = self.config.get('width', None)  # console width for rich table displays of synapse measures
        validation_params = (random_uids, query_responses, return_ops, times, routing_score,
                             inputs, val_len, self.loss_fct,
                             self.config.nucleus.scaling_law_power, self.config.nucleus.synergy_scaling_law_power,
                             console_width, self.config.logging.debug or self.config.logging.trace)

        loss = torch.tensor(0.).to(self.device)  # to accumulate neuron_loss and routing_loss over synapses
        neuron_stats = {}  # to gather neuron synapse validation measures and statistics

        # === Validate synapse responses ===
        # Iterate over all queried synapses and validate responses
        for i, (synapse, validate_func) in enumerate(synapses):
            _loss, stats = validate_func(*validation_params, synapse=synapse, index_s=i)  # validate individual synapse
            loss += _loss  # add neuron_loss and routing_loss

            for _uid, _stats in stats.items():
                neuron_stats.setdefault(_uid, {})
                neuron_stats[_uid].update(_stats)  # gather neuron synapse validation measures and statistics

        return loss, neuron_stats




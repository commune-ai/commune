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

import os
import time
import math
import copy
import queue
import torch
import random
import bittensor
import argparse
import bittensor as bt
import traceback
import commune as c

from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from transformers import AutoTokenizer
from datasets import load_dataset



class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )


    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 1 )
        parser.add_argument( '--neuron.device', type = str, help = 'Device to run the validator on.', default = "cuda" if torch.cuda.is_available() else "cpu" )
        parser.add_argument( '--neuron.epoch_length_override', type = int, help = 'Override the default timeout', default = -1 )
        parser.add_argument( '--neuron.max_network_delay', type=int, default = 20 )
        parser.add_argument( '--neuron.topk', type=int, default = 50 )
        parser.add_argument( '--neuron.alpha', type=float, default = 0.5 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        bt.axon.add_args( parser )
        cls.add_args( parser )
        return bt.config( parser )
    
    def __init__( self, config = None, subtensor = None, wallet = None, netuid=None):      
        self.config = neuron.config() if config == None else config


        self.check_config( self.config )
        bt.logging( config = self.config )
        print( self.config )
        
        self.subtensor = bt.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.device = torch.device( self.config.neuron.device )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.hotkeys = self.metagraph.hotkeys

        if wallet == None:
            wallet = bt.wallet ( config = self.config ) if wallet == None else wallet
            wallet.create_if_non_existent()
            wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        
        self.wallet = wallet
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )
        self.sync()
        
        
    def sync(self):
        self.metagraph.sync()
        self.last_sync = self.subtensor.block
        self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.top_uids = self.get_top_uids()

    def get_top_uids(self, k=None) -> Dict:
        if k == None:
            k = self.config.neuron.topk
        if not hasattr(self, 'bt'):
            self.bt = c.module('bittensor')
        
        top_uids = self.bt.get_top_uids(k=k, metagraph=self.metagraph, return_dict=True)
        
        return top_uids

    
    def run( self ):
        """ Training 
            The function uses an infinite loop to repeatedly generate a random question, 
            ask the network to complete the question, and train the gating network using 
            the question and the resulting completions.
        """
        # Store the current epoch block number for comparison later.
        last_epoch_block = self.subtensor.block
        self.last_sync = self.subtensor.block
        weights_set_for_epoch = False
        # Start an infinite loop for training.
        last_epoch_set_block = 0
        current_block = 0
        while True:
            if current_block == self.subtensor.block:
                continue
            else:
                current_block = self.subtensor.block
            
            # Resync metagraph before returning. (sync every 15 min or ~75 blocks)
            if self.subtensor.block - self.last_sync > self.config.neuron.max_network_delay:
                self.sync()
                
                
            # Check if enough epoch blocks have elapsed since the last epoch.
            epoch_length = self.subtensor.validator_epoch_length(self.config.netuid) if self.config.neuron.epoch_length_override == -1 else self.config.neuron.epoch_length_override
            blocks_until_epoch = epoch_length - ( self.subtensor.block - last_epoch_block )
            bittensor.logging.debug( 'blocks_until_epoch', blocks_until_epoch )

            
            bittensor.logging.info( 'block', self.subtensor.block )
            bittensor.logging.info( f'blocks until epoch {blocks_until_epoch} (epoch_length: {epoch_length})' )

            if blocks_until_epoch % (epoch_length//2) == 0  : 
                bittensor.logging.trace( f'epoch FINITO({epoch_length})' )

                # Update the last epoch block to the current epoch block.
                last_epoch_block = self.subtensor.block
                
                # Computes the average reward for each uid across non-zero values 
                # using the rewards history stored in the self.history list.
                uids, weights = self.compute_weights()
                bittensor.logging.info( 'weights', weights )
                # Set the weights on chain via our subtensor connection.
                
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = uids,
                    weights = weights,
                    wait_for_finalization = False,
                )
                last_epoch_set_block = self.subtensor.block
                weights_set_for_epoch = True


    def compute_weights( self ) -> Tuple[ torch.LongTensor, torch.FloatTensor ]:
        """
            Computes the average reward for each uid across non-zero values 
            using the rewards history stored in the self.history list.

            Returns:
                uids ( torch.LongTensor, shape = (n) ): 
                    Uid to set weights on.
                weights ( torch.FloatTensor, shape = (n) ): 
                    The weights for each uid.
        """
        bittensor.logging.info( 'compute_weights()' )

        self.sync()
        
        uids = torch.tensor(list(map(int, self.top_uids.keys()))).long().to('cpu')
        weights = torch.tensor(list(map(int, self.top_uids.values()))) + 1e-10
        weights = weights / (weights.sum()).to('cpu')
     
        # # Process the raw weights to final_weights via subtensor limitations.
        # processed_weight_uids, processed_weights = bittensor.utils.weight_utils.process_weights_for_netuid(
        #     uids = uids.to( "cpu" ),
        #     weights = weights.to( "cpu" ),
        #     netuid = self.config.netuid,
        #     subtensor = self.subtensor,
        #     metagraph = self.metagraph
        # )
        # bittensor.logging.trace( 'processed_weights', processed_weights )
        # bittensor.logging.trace( 'processed_weight_uids', processed_weight_uids )
        return uids, weights

if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().run()

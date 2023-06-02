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

from loguru import logger
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from reward import RewardModel
from gating import GatingModel
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
        parser.add_argument( '--neuron.topk', type=int, default = 100 )
        parser.add_argument( '--neuron.alpha', type=float, default = 0.5 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.logging.add_args( parser )
        bt.axon.add_args( parser )
        GatingModel.add_args( parser )
        cls.add_args( parser )
        return bt.config( parser )
    
    def __init__( self, config = None, subtensor = None, wallet = None, bt=None):      
        self.config = neuron.config() if config == None else config


        self.check_config( self.config )
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        print( self.config )
        
        self.subtensor = bt.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.device = torch.device( self.config.neuron.device )
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        
        if wallet == None:
            wallet = bt.wallet ( config = self.config ) if wallet == None else wallet
            wallet.create_if_non_existent()
            wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        
        self.wallet = wallet
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )
        self.hotkeys = self.metagraph.hotkeys

        self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        # History of forward events.
        # Get a list of peers delegating to me
        delegated = self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )
        self.my_nominators = { nomin[0]: nomin[1] for nomin in delegated[0][0].nominators } if len(delegated) else {}
      
    def sync(self):
        self.metagraph.sync()
        self.last_sync = self.subtensor.block
        self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.my_nominators = { nomin[0]: nomin[1] for nomin in delegates[0][0].nominators } if len(delegates) else {}
        self.top_uids = self.update_top_uids()
    top_uids = {}
    def update_top_uids(self, k=None)  
        if k == None:
            k = self.config.neuron.topk
        if not hasattr(self, 'bt'):
            self.bt = c.module('bittensor')
        
        top_uids = self.bt.get_top_neurons(k=k, metagraph=self.metagraph, return_dict=True)
        
        for uid, incentive in top_uids.items():
            self.top_uids[uid] = self.top_uids.get(uid, incentive) * self.config.neuron.alpha + incentive * (1-self.config.neuron.alpha)
        
        return self.top_uids

    
    def run( self ):
        """ Training 
            The function uses an infinite loop to repeatedly generate a random question, 
            ask the network to complete the question, and train the gating network using 
            the question and the resulting completions.
        """
        # Store the current epoch block number for comparison later.
        last_epoch_block = self.subtensor.block
        self.last_sync = self.subtensor.block
        
        # Start an infinite loop for training.
        while True:

            # Resync metagraph before returning. (sync every 15 min or ~75 blocks)
            if self.subtensor.block - self.last_sync > self.config.neuron.max_network_delay:
                self.sync()
                
                
            # Check if enough epoch blocks have elapsed since the last epoch.
            epoch_length = self.subtensor.validator_epoch_length(self.config.netuid) if self.config.neuron.epoch_length_override == -1 else self.config.neuron.epoch_length_override
            blocks_until_epoch = epoch_length - ( self.subtensor.block - last_epoch_block )
            bittensor.logging.debug( 'blocks_until_epoch', blocks_until_epoch )
            

            if blocks_until_epoch <= 0: 
                bittensor.logging.trace( 'epoch()' )
                bittensor.logging.info( 'block', self.subtensor.block )

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
            steps += 1 


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

        if len(self.top_uids) == 0:
            self.sync()
        
        uids = torch.tensor(list(map(int, self.top_uids.keys()))).long()
        weights = torch.tensor(list(map(int, self.top_uids.values()))) 
        weights = weights / (weights.sum() + 1e-10)
     
        # Process the raw weights to final_weights via subtensor limitations.
        processed_weight_uids, processed_weights = bittensor.utils.weight_utils.process_weights_for_netuid(
            uids = uids.to( "cpu" ),
            weights = weights.to( "cpu" ),
            netuid = self.config.netuid,
            subtensor = self.subtensor,
            metagraph = self.metagraph
        )
        bittensor.logging.trace( 'processed_weights', processed_weights )
        bittensor.logging.trace( 'processed_weight_uids', processed_weight_uids )
        return processed_weight_uids, processed_weights

    def run(self):
            # Start an infinite loop, allows axon to service inference requests.
            last_sync = self.subtensor.block
            while True:
                time.sleep(12)
                if self.subtensor.block -last_sync > 100:
                    

        else:
            # Normal validator train operation for validation.
            self.train()

if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().run()

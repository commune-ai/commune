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

__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific. Do not include the answer in the question.
'''

__default_base_prompt__ = '''
You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
'''

__default_follow_up_prompt__ = '''
Ask a follow up question.
'''
class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )

        if not config.neuron.dont_save_events:
            # Add custom event logger for the events.
            logger.level("EVENTS", no=38, icon="📝")
            logger.add( 
                config.neuron.full_path + "/" + "completions.log", 
                rotation=config.neuron.events_retention_size, serialize=True, enqueue=True, backtrace=False, diagnose=False, level="EVENTS", 
                format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message} | {extra[prompt]} {extra[completion]} {extra[uids]} {extra[all_uids]} {extra[rewards]}{extra[all_completions]} {extra[block]}"
            )

    def record_event( self, event: SimpleNamespace ):
        self.history.put( event )
        if not self.config.neuron.dont_save_events:
            logger.log(
                "EVENTS", 
                "events", 
                prompt = event.message,
                completion = event.completion,
                uids = event.uids.tolist(),
                all_uids = event.all_uids.tolist(),
                rewards = event.rewards.tolist(),
                all_completions = event.all_completions,
                block = event.block.item(),
            )

    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 1 )
        parser.add_argument( '--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument( '--neuron.base_prompt', type=str, help = 'Prompt injected before a question is completed by miners on the network', default = __default_base_prompt__ )
        parser.add_argument( '--neuron.follow_up_prompt', type=str, help = 'Follow up prompt that is completed by miners on the network.', default = __default_follow_up_prompt__ )
        parser.add_argument( '--neuron.reset_bootstrap_prompt_frequency', type=int, help = 'How frequent to use the base follow up question.', default = 3 )
        parser.add_argument( '--neuron.question_prompt', type=str, help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = __default_question_prompt__ )
        parser.add_argument( '--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument( '--neuron.length_timeout_multiplier', type = int, help = 'Base timeout for all requests.', default = 0.01 )
        parser.add_argument( '--neuron.inference_topk', type = int, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument( '--neuron.training_topk', type = int, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 50 )
        parser.add_argument( '--neuron.training_timeout', type = int, help = 'Query timeout during training', default = 4 )
        parser.add_argument( '--neuron.inference_timeout', type = int, help = 'Query timeout during inference', default = 10 )
        parser.add_argument( '--neuron.inference_only', action = 'store_true', help = 'If set, training off and only inference will be served via axon.', default = False )
        parser.add_argument( '--neuron.reward_path', type = str, help = 'Path to reward model.', default = '~/.bittensor/reward_models' )
        parser.add_argument( '--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 100000 )
        parser.add_argument( '--neuron.device', type = str, help = 'Device to run the validator on.', default = "cuda" if torch.cuda.is_available() else "cpu" )
        parser.add_argument( '--neuron.epoch_length_override', type = int, help = 'Override the default timeout', default = -1 )
        parser.add_argument( '--neuron.dont_save_events', action = 'store_true', help = 'If set, we dont save events to a log file.', default = False )
        parser.add_argument( '--neuron.events_retention_size',  type = str,  help = 'Events retention size.', default = "2 GB" )
        parser.add_argument( '--neuron.no_reward_model', action = 'store_true', help = 'If set, we dont load the reward model instead use just the scores.', default = False )
        parser.add_argument( '--neuron.question_random_sample_uids', action = 'store_true', help = 'If set, random sample uids to get question.', default = False )
        parser.add_argument( '--neuron.reward_shift', type = int, help = 'The value to shift rewards for calculation.', default = 3 )

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

        if bt == None:
            self.top_uids = c.module('bittensor').get_top_neurons(t=100)
        self.check_config( self.config )
        bt.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        print( self.config )
        
        self.subtensor = bt.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.device = torch.device( self.config.neuron.device )
        self.wallet = bt.wallet ( config = self.config ) if wallet == None else wallet
        self.metagraph = bt.metagraph( netuid = self.config.netuid, network = self.subtensor.network )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )
        self.tokenizer = AutoTokenizer.from_pretrained( 'EleutherAI/gpt-j-6b' )

        # check if invoking iter() is indeed necessary
        self.dataset = iter(load_dataset('squad_v2', split='train', streaming=True).shuffle(buffer_size=10000))

        self.moving_averaged_scores = torch.zeros((self.metagraph.n)).to( self.device )
        self.alpha = 0.99
        self.hotkeys = self.metagraph.hotkeys

        self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
        # History of forward events.
        # Get a list of peers delegating to me
        delegated = self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )
        self.my_nominators = { nomin[0]: nomin[1] for nomin in delegated[0][0].nominators } if len(delegated) else {}

        self.load()
        self.check_weights()

    def train( self ):
        """ Training 
            The function uses an infinite loop to repeatedly generate a random question, 
            ask the network to complete the question, and train the gating network using 
            the question and the resulting completions.
        """
        # Store the current epoch block number for comparison later.
        last_epoch_block = self.subtensor.block
        steps = 0
        
        self.base_prompt = self.config.neuron.base_prompt
        reward_diff = 0
        self.last_sync = self.subtensor.block
        
        # Start an infinite loop for training.
        try:
            while True:

                # Resync metagraph before returning. (sync every 15 min or ~75 blocks)
                if self.subtensor.block - self.last_sync > 100:
                    self.metagraph.sync()
                    self.last_sync = self.subtensor.block

                    delegates = self.subtensor.get_delegated( self.wallet.coldkeypub.ss58_address )

                    # Recreate pools here to ensure sizing is correct.
                    self.dendrite_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )
                    self.inference_pool = bt.text_prompting_pool( keypair = self.wallet.hotkey, metagraph = self.metagraph )

                    self.my_nominators = { nomin[0]: nomin[1] for nomin in delegates[0][0].nominators } if len(delegates) else {}

                    if self.metagraph.n > self.gating_model.num_uids:
                        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to( self.device )

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
                        uids = self.top_uids,
                        weights = torch.ones((len(self.top_uids))).to(torch.float32),
                        wait_for_finalization = False,
                    )
                steps += 1 

        except Exception as e:
            bittensor.logging.info( 'Error in training loop', str( e    ) )
            print(traceback.format_exc())
    
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


        uids = self.top_uids
        weights = torch.nn.functional.normalize( torch.ones(), p=1, dim=0 )
     
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
        if self.config.neuron.inference_only:
            # Start an infinite loop, allows axon to service inference requests.
            last_sync = self.subtensor.block
            while True:
                time.sleep(12)
                if self.subtensor.block -last_sync > 100:
                    self.metagraph.sync()
                    self.last_sync = self.subtensor.block
                    self.load(inference_only = True)

        else:
            # Normal validator train operation for validation.
            self.train()

if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().run()

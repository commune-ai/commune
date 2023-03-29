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
from nucleus import nucleus, stats_table, scaling_law_loss_to_params
import sys, os
logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

# Neuron stats recorded by validator neuron/nucleus
#   [Column_name, key_name, format_string, rich_style]  # description

class neuron:
    r"""
    Creates a bittensor neuron that specializes validating other peers. The core validator
    finetunes on the bittensor network with a mixture of experts model and shapely scoring.
    The validator's main jobs are to identify important/useful peers in the network and correctly
    weight them. To achieve this, the validator will send requests to different peers on the network
    and evalute their responses.

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            dataset (:obj:bittensor.dataset , `optional`):
                bittensor dataset 
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            dendrite (:obj:bittensor.dendrite, `optional`):
                bittensor dendrite object
            dataset (:obj:bittensor.dendrite, `optional`):
                bittensor dendrite object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> validator = bittensor.neuron.text.core_validator.neuron(subtensor=subtensor)
            >>> validator.run()
    """
    def __init__( 
        self, 
        config: 'bittensor.Config' = None,
        wallet: 'bittensor.Wallet' = None,
        subtensor: 'bittensor.Subtensor' = None,
        metagraph: 'bittensor.Metagraph' = None,
        dendrite: 'bittensor.Dendrite' = None,
        dataset: 'bittensor.dataset' = None,
        axon: 'bittensor.axon' = None
    ):

        # === Set up Config ===

        config = config if config else self.config()
        # if config == None: config = neuron.config()
        self.config = config
        neuron.check_config( self.config )
        self.config.to_defaults()
        
        # ===  Logging + prometheus ===
        self.config.to_prometheus()
        bittensor.logging( 
            config = self.config, 
            logging_dir = self.config.neuron.full_path 
        )
        bittensor.prometheus ( 
            config = self.config, 
            port = config.prometheus.port if config.axon.port == bittensor.defaults.axon.port else config.axon.port - 1000
        )

        # === Create Bittensor objects ===
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.wallet = bittensor.wallet ( config = self.config ) if wallet == None else wallet
        self.subtensor = bittensor.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor ) if metagraph == None else metagraph
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet, max_active_receptors = 0 ) if dendrite == None else dendrite # Dendrite should not store receptor in validator.
        # self.axon = bittensor.axon ( config = self.config, wallet = self.wallet ) if axon == None else axon
        self.device = torch.device ( device = self.config.neuron.device )    
        self.nucleus = nucleus ( config = self.config, device = self.device, subtensor = self.subtensor ).to( self.device )
        self.dataset = (bittensor.dataset(config=self.config, batch_size=self.subtensor.validator_batch_size,
                                          block_size=self.subtensor.validator_sequence_length + self.config.neuron.validation_len)
                        if dataset is None else dataset)
        self.optimizer = torch.optim.SGD(
            self.nucleus.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum
        )

        # === Create thread queue ===
        self.loss = None
        self.loss_agg_mutex = Lock()

        # === Neuron statistics variables ===
        self.neuron_stats = {}  # neuron statistics dict of dicts: [uid] -> {'stat1': val1, 'stat2': val2, ...}
        self.neuron_hotkeys = []  # keep neuron hotkeys to compare and check for changes after metagraph.sync()
        self.neuron_changes = {}  # neuron hotkey changes dict of dicts of dicts: [uid] -> [block] -> {'new_hotkey': , 'old_hotkey': , 'old_stats':}
        self.alpha = 0.1  # EMA coefficient in [0, 1], higher alpha discounts older observations faster


        if self.config.neuron.validation_synapse == 'TextCausalLMNext':
            self.weight_key = 'shapley_values_nxt'  # stat key + ! to calculate neuron weights with
            # stat keys to duplicate (['key']->['key!']) and push zero to its EMA if neuron non-responsive
            self.synapse_keys = ['shapley_values_nxt']
        else:
            self.weight_key = 'shapley_values_min'  # stat key + ! to calculate neuron weights with
            # stat keys to duplicate (['key']->['key!']) and push zero to its EMA if neuron non-responsive
            self.synapse_keys = ['shapley_values_min']

        # === Prometheus stats ===
        # Turn this off by passing the --prometheus.off flag
        self.prometheus_info = Info("neuron_info", "Info sumamries for the running server-miner.")
        self.prometheus_gauges = Gauge('validator_gauges', 'Gauges for the running validator.', ['validator_gauges_name'])
        self.prometheus_counters = Counter('validator_counters', 'Counters for the running validator.', ['validator_counters_name'])
        self.prometheus_step_time = Histogram('validator_step_time', 'Validator step time histogram.', buckets=list(range(0,2*bittensor.__blocktime__,1)))

        # load last saved validator values from the file system
        if not config.neuron.restart:
            self.load()

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        nucleus.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.prometheus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_validator')
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.1 )
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8 )
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch, -1 value means we use the chain value.', default = -1 )
        parser.add_argument('--neuron.epochs_until_reset', type=int, help='Number of epochs before weights are reset.', default = -1 )
        parser.add_argument('--neuron.validation_len', type=int, help='Number of tokens to holdout for phrase validation beyond sequence context.', default=8)
        parser.add_argument('--neuron.prune_len', type=int, help='Number of tokens to prune from each validation input sequence.', default=1)
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0 )
        parser.add_argument('--neuron.track_hotkey_changes', action='store_true', help='If True, track hotkey changes.', default=False)
        parser.add_argument('--neuron.restart', action='store_true', help='If True, reset neuron_stats and validate anew.', default=False)
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )
        parser.add_argument('--neuron.wait_for_finalization', action='store_true', help='''when setting weights the miner waits for trnasaction finalization.''', default=False)
        parser.add_argument('--neuron.forward_num', type=int, help='''How much forward request before a backward call.''', default=3)
        parser.add_argument('--neuron.validation_synapse', type=str, help='''Synapse used for validation.''', default='TextCausalLMNext', choices = ['TextCausalLMNext', 'TextCausalLM'])
        parser.add_argument('--neuron.exclude_quantile', type=float, help='Exclude the lowest quantile from weight setting. (default value: -1, pulling from subtensor directly)', default=-1)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )        
        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.axon.add_args( parser )
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (f'[bold]UID {self.uid}[/bold] \[{self.dendrite.receptor_pool.external_ip}] '
                f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/'
                f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])')

    def __del__(self):
        self.dataset.close()
        self.dendrite.__del__()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        r""" Close down neuron.
        """
        print(exc_type, exc_value, exc_traceback)
        self.__del__()

    def __enter__(self):
        r""" Sanity checks and begin validator.
        """
        # === Wallet ===
        # Connects wallet to network. 
        self.wallet.create()
        # NOTE: This registration step should likely be solved offline first.
        self.wallet.reregister( subtensor = self.subtensor )


        # === UID ===
        # Get our uid from the chain. 
        # At this point we should have a uid because we are already registered.
        self.uid = self.wallet.get_uid( subtensor = self.subtensor )    

        # === Monitoring ===
        # Optionally set up wandb logging.
        if self.config.using_wandb:
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )

        # === Set prometheus run info ===
        # Serve the axon so we can determine where the prometheus server port is (the axon is only served for this reason.)
        # self.axon.serve( subtensor = self.subtensor )
        self.prometheus_gauges.labels( "model_size_params" ).set( sum(p.numel() for p in self.nucleus.parameters()) )
        self.prometheus_gauges.labels( "model_size_bytes" ).set( sum(p.element_size() * p.nelement() for p in self.nucleus.parameters()) )
        self.prometheus_info.info({
            'type': "core_validator",
            'uid': str(self.uid),
            'network': self.config.subtensor.network,
            'coldkey': str(self.wallet.coldkeypub.ss58_address),
            'hotkey': str(self.wallet.hotkey.ss58_address),
        })

    def save(self, path=None):
        r""" Save validated hotkeys and neuron_stats to filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path

            state_dict = {
                'neuron_stats': self.neuron_stats,
                'neuron_hotkeys': self.neuron_hotkeys
            }

            if self.config.neuron.track_hotkey_changes:
                state_dict['neuron_changes'] = self.neuron_changes

            torch.save(state_dict, f'{path}/model.torch')
            bittensor.logging.success(prefix='Saved model', sufix=f'<blue>{path}/model.torch</blue>')

        except Exception as e:
            logger.warning(f'Failed to save model with error: {e}')

    def load(self, path=None):
        r""" Load validated hotkeys and neuron_stats from filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path
            state_dict = torch.load(f'{path}/model.torch')

            self.neuron_stats = state_dict['neuron_stats']
            self.neuron_hotkeys = state_dict['neuron_hotkeys']

            if 'neuron_changes' in state_dict and self.config.neuron.track_hotkey_changes:
                self.neuron_changes = state_dict['neuron_changes']

            bittensor.logging.success(prefix='Reloaded model', sufix=f'<blue>{path}/model.torch</blue>')

        except Exception as e:
            logger.warning(f'Failed to load model with error: {e}')

    def run ( self ):
        r""" Run the validator and terminate on Keyboard interrupt.
        """
        # === Setup ===
        # Checks wallet and starts monitoring.
        with self:

            # === Start forward requests ===
            self.metagraph_sync()
            
            # === Run ===
            # Iterates through epochs.
            self.epoch = 0
            self.global_step = 0
            while True:
                try:

                    # === Epoch ===
                    # Each epoch runs for blocks_per_epoch and resets
                    # the model every epochs_until_reset.
                    self.run_epoch()

                # === Stops on interrupt otherwise restarts ===
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.prometheus_counters.labels('failures').inc()
                    console.print_exception(show_locals=False)
                    print( traceback.format_exc() )
                    print( 'Unknown exception: {}', e )
                    if not self.config.neuron.restart_on_failure:
                        break

    def run_epoch( self ):
        r""" Runs a validator epoch. We apply batches until the epoch length is exhausted.
            Occasionally the validator nucleus is completely reset to ensure we dont converge to far.
            At the end of the epoch we set weights on the chain and optionally log to wandb.
        """
        # === Get params for epoch ===
        # Pulling the latest chain parameters.
        current_block = self.subtensor.block
        batch_size = self.subtensor.validator_batch_size 
        sequence_length = self.subtensor.validator_sequence_length
        validation_len = self.config.neuron.validation_len  # Number of tokens to holdout for phrase validation beyond sequence context
        prune_len = self.config.neuron.prune_len  # Number of tokens to holdout for phrase validation beyond sequence context
        min_allowed_weights = self.subtensor.min_allowed_weights
        max_weight_limit = self.subtensor.max_weight_limit
        blocks_per_epoch = self.subtensor.validator_epoch_length if self.config.neuron.blocks_per_epoch == -1 else self.config.neuron.blocks_per_epoch
        epochs_until_reset = self.subtensor.validator_epochs_per_reset if self.config.neuron.epochs_until_reset == -1 else self.config.neuron.epochs_until_reset
        self.config.nucleus.scaling_law_power = self.subtensor.scaling_law_power
        self.config.nucleus.synergy_scaling_law_power = self.subtensor.synergy_scaling_law_power

        # === Logs Prometheus ===
        self.prometheus_gauges.labels("current_block").set( current_block )
        self.prometheus_gauges.labels("batch_size").set( batch_size )
        self.prometheus_gauges.labels("sequence_length").set( sequence_length )
        self.prometheus_gauges.labels("validation_len").set( validation_len )
        self.prometheus_gauges.labels("min_allowed_weights").set( min_allowed_weights )
        self.prometheus_gauges.labels("blocks_per_epoch").set( blocks_per_epoch )
        self.prometheus_gauges.labels("epochs_until_reset").set( epochs_until_reset )
        self.prometheus_gauges.labels("scaling_law_power").set( self.config.nucleus.scaling_law_power )
        self.prometheus_gauges.labels("synergy_scaling_law_power").set( self.config.nucleus.synergy_scaling_law_power )

        # === Update dataset size ===
        if (batch_size != self.dataset.batch_size) or (sequence_length + validation_len + prune_len != self.dataset.block_size):
            self.dataset.set_data_size(batch_size, sequence_length + validation_len + prune_len)

        # === Logs ===
        if self.config.using_wandb:
            wandb.log({'era/batch_size': batch_size, 'era/sequence_length': sequence_length,
                       'era/validation_len': validation_len,
                       'era/min_allowed_weights': min_allowed_weights, 'era/max_weight_limit': max_weight_limit,
                       'era/blocks_per_epoch': blocks_per_epoch, 'era/epochs_until_reset': epochs_until_reset},
                      step=current_block)

        # === Run Epoch ===
        # Each block length lasts blocks_per_epoch blocks.
        # This gives us a consistent network wide timer.
        # Here we run until blocks_per_epochs have progressed.

        epoch_steps = 0
        epoch_responsive_uids = set()
        epoch_queried_uids = set()
        epoch_start_time = time.time()

        self.prometheus_gauges.labels("epoch_steps").set(0)

        # normal epoch duration is blocks_per_epoch if all UIDs have been queried
        # try to query each UID at least once - assumes nucleus samples without replacement
        # but keep minimum epoch duration at blocks_per_epoch * block_period
        # in case of subtensor outage causing invalid block readings to prevent fast repeated weight setting
        start_block = self.subtensor.block
        while (self.subtensor.block < start_block + blocks_per_epoch or
               time.time() - epoch_start_time < blocks_per_epoch * bittensor.__blocktime__):

            logger.info(f'Run epoch {self.epoch} (step {epoch_steps}) while '
                        f'({self.subtensor.block} < {start_block + blocks_per_epoch} '
                        f'= {start_block} + {blocks_per_epoch}) or '
                        f'({time.time() - epoch_start_time:.2f} < {blocks_per_epoch * bittensor.__blocktime__})')

            start_time = time.time()

            # === Forward ===
            # Forwards inputs through the network and returns the loss
            # and endpoint scores using shapely approximation of salience.
            loss, stats = self.nucleus( next(self.dataset) , self.metagraph, self.dendrite )
            self.prometheus_gauges.labels("loss").set( loss.item() )

            # === Backward ===
            # Backwards gradients through model to train gating and remote endpoints.
            if hasattr(loss, 'grad_fn') and loss.grad_fn is not None:
                logger.info(f'Backward <dim>(loss: {loss:.3f})</dim>')
                bw_start_time = time.time()
                (loss / self.config.neuron.forward_num).backward()
                logger.info(f'Backward <dim>[{time.time() - bw_start_time:.3g}s]</dim>')

            # === Stats update ===
            # Updates moving averages and history.
            responsive_uids, queried_uids = self.neuron_stats_update(stats)

            epoch_responsive_uids |= set(responsive_uids)
            epoch_queried_uids |= set(queried_uids)

            # === State update ===
            # Prints step logs to screen.
            epoch_steps += 1
            self.global_step += 1
            self.prometheus_gauges.labels("global_step").inc()
            self.prometheus_gauges.labels("epoch_steps").inc()

            # === Block state ===
            current_block = self.subtensor.block
            self.prometheus_gauges.labels("current_block").set(current_block)
            self.prometheus_gauges.labels("last_updated").set( current_block - self.metagraph.last_update[self.uid] )

            # === Step time ===
            step_time = time.time() - start_time
            self.prometheus_step_time.observe( step_time )
            self.prometheus_gauges.labels('step_time').set( step_time )
            
            if epoch_steps % 25 == 1:
                # validator identifier status console message (every 25 validation steps)
                print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                      f"{f'[bright_white]core_validator[/bright_white]'.center(16 + len('[bright_white][/bright_white]'))} | "
                      f"UID [cyan]{self.uid}[/cyan] "
                      f"[dim white not bold][{self.dendrite.receptor_pool.external_ip}][/dim white not bold] "
                      f"[white not bold]cold:[bold]{self.wallet.name}[/bold]:"
                      f"[bright_white not bold]{self.wallet.coldkeypub.ss58_address}[/bright_white not bold] "
                      f"[dim white]/[/dim white] "
                      f"hot:[bold]{self.config.wallet.hotkey}[/bold]:"
                      f"[bright_white not bold]{self.wallet.hotkey.ss58_address}[/bright_white not bold][/white not bold]")

                # validator update status console message
                print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                      f"{f'UID [bright_cyan]{self.uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
                      f'Updated [yellow]{current_block - self.metagraph.last_update[self.uid]}[/yellow] [dim]blocks ago[/dim] | '
                      f'Dividends [green not bold]{self.metagraph.dividends[self.uid]:.5f}[/green not bold] | '
                      f'Stake \u03C4[magenta not bold]{self.metagraph.stake[self.uid]:.5f}[/magenta not bold] '
                      f'[dim](retrieved [yellow]{current_block - start_block}[/yellow] blocks ago from {self.subtensor.network})[/dim]')

                # save neuron_stats to filesystem
                self.save()

            # step update console message (every validation step)
            print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                  f"{f'[magenta dim not bold]#{current_block}[/magenta dim not bold]'.center(16 + len('[magenta dim not bold][/magenta dim not bold]'))} | "
                  f'[green not bold]{current_block - start_block}[/green not bold]/'
                  f'[white not bold]{blocks_per_epoch}[/white not bold] [dim]blocks/epoch[/dim] | '
                  f'[white not bold]Step {epoch_steps}[white not bold] '
                  f'[dim] Epoch {self.epoch}[/dim] | '
                  f'[bright_green not bold]{len(responsive_uids)}[/bright_green not bold]/'
                  f'[white]{len(queried_uids)}[/white] '
                  f'[[yellow]{step_time:.3g}[/yellow]s] '
                  f'[dim white not bold][green]{len(epoch_responsive_uids)}[/green]/'
                  f'{len(epoch_queried_uids)}[/dim white not bold]')

            if self.config.logging.debug or self.config.logging.trace:
                # === Print stats update (table) ===
                # Prints exponential moving average statistics of valid neurons from latest validator forward
                stats_table({uid: self.neuron_stats[uid]
                             for uid, stat in stats.items() if len(set(stat.keys()) & set(self.synapse_keys))},
                            self.weight_key, self.config.get('width', None),
                            f'[white] Stats update [/white] | ' + str(self),  # title
                            f'#{current_block}: '
                            f'[bold]{current_block - start_block}[/bold]/{blocks_per_epoch} (blocks/epoch) | '
                            f'Epoch {self.epoch} | '
                            f'[white] Step {epoch_steps} ({self.global_step} global) \[{step_time:.3g}s] [/white]')  # caption

                # === Calculate neuron weights ===
                sample_uids, sample_weights = self.calculate_weights()
                self.weights_table(sample_uids, sample_weights,
                                   include_uids=list(stats.keys()), num_rows=len(stats) + 25)  # print weights table

            # === Logs ===
            if self.config.using_wandb:
                for uid, vals in self.neuron_stats.items():
                    for key in vals:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                        wandb.log({f'stats/{key}_{uid}': vals[key]}, step=current_block, commit=False)

                wandb.log({'epoch/epoch': self.epoch, 'epoch/epoch_steps': epoch_steps,
                           'epoch/global_steps': self.global_step, 'epoch/loss': loss.item(),
                           'epoch/time': step_time}, step=current_block, commit=True)

            # Do the backward request after the a queue of forward requests got finished.  
            if epoch_steps % self.config.neuron.forward_num == 1:
                start_time = time.time()
                logger.info('Model update \t| Optimizer step')

                # === Apply gradients ===
                # Applies local gradients to parameters.
                clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
                self.optimizer.step()
                self.optimizer.zero_grad()
                logger.info(f'Model update \t| Optimizer step <dim>[{time.time() - start_time:.3g}s]</dim>')

        self.metagraph_sync()  # Reset metagraph.

        # === Calculate neuron weights ===
        sample_uids, sample_weights = self.calculate_weights()

        if self.config.logging.debug or self.config.logging.trace:
            self.weights_table(sample_uids, sample_weights)  # print weights table

        # set weights console message (every epoch)
        print(f"[white not bold]{datetime.datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
              f"{f'[bright_white]Set weights[/bright_white]'.center(16 + len('[bright_white][/bright_white]'))} | "
              f'[bright_green not bold]{len(sample_weights)}[/bright_green not bold] [dim]weights set[/dim] | '
              f'[bright_green not bold]{len(epoch_responsive_uids)}[/bright_green not bold]/'
              f'[white]{len(epoch_queried_uids)}[/white] '
              f'[dim white not bold][green]responsive[/green]/queried[/dim white not bold] '
              f'[[yellow]{time.time() - epoch_start_time:.0f}[/yellow]s] | '
              f'[dim]weights[/dim] sum:{sample_weights.sum().item():.2g} '
              f'[white] max:[bold]{sample_weights.max().item():.4g}[/bold] / '
              f'min:[bold]{sample_weights.min().item():.4g}[/bold] [/white] '
              f'\[{max_weight_limit:.4g} allowed]')

        self.subtensor.set_weights(
            uids=sample_uids.detach().to('cpu'),
            weights=sample_weights.detach().to('cpu'),
            wallet=self.wallet,
            wait_for_finalization=self.config.neuron.wait_for_finalization,
        )

        # === Wandb Logs ===
        # Optionally send validator logs to wandb.
        if self.config.using_wandb:
            # Logging history to wandb.
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'weights', index = sample_uids, values = torch.zeros( self.metagraph.n ).scatter( dim = 0, src = sample_weights, index = sample_uids ) ),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1); df['uid'] = df.index
            wandb_data_dend = self.dendrite.to_wandb()
            wandb_weight = {f'stats/weight_{uid}': weight for uid, weight in zip (sample_uids, sample_weights)}
            wandb_data = { 'stake': self.metagraph.S[ self.uid ].item(), 'dividends': self.metagraph.D[ self.uid ].item() } 
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block, commit=False)
            wandb.log( { **wandb_data, **wandb_data_dend, **wandb_weight }, step = current_block, commit=True)

        # === Epoch Prometheus ===
        self.prometheus_gauges.labels("epoch").inc()
        self.prometheus_gauges.labels("set_weights").inc()
        self.prometheus_gauges.labels("stake").set( self.metagraph.stake[self.uid] )
        self.prometheus_gauges.labels("rank").set( self.metagraph.ranks[self.uid] )
        self.prometheus_gauges.labels("trust").set( self.metagraph.trust[self.uid] )
        self.prometheus_gauges.labels("incentive").set( self.metagraph.incentive[self.uid] )
        self.prometheus_gauges.labels("dividends").set( self.metagraph.dividends[self.uid] )
        self.prometheus_gauges.labels("emission").set( self.metagraph.emission[self.uid] )

        # Iterate epochs.
        self.epoch += 1

    def metagraph_sync(self):
        r""" Syncing metagraph together with other metagraph-size related objects
        """
        old_hotkeys = self.neuron_hotkeys + [] if self.neuron_hotkeys else self.metagraph.hotkeys
        self.metagraph.sync()
        self.neuron_hotkeys = self.metagraph.hotkeys

        changed_hotkeys = []
        # === Reset neuron stats if uid got replaced
        for uid, old_hotkey in enumerate(old_hotkeys):
            if old_hotkey != self.neuron_hotkeys[uid]:
                if self.config.neuron.track_hotkey_changes:
                    block = self.subtensor.block
                    self.neuron_changes.setdefault(uid, {})  # [uid] -> dict() of blocks
                    self.neuron_changes[uid][block] = {'new_hotkey': self.neuron_hotkeys[uid], 'old_hotkey': old_hotkey}
                    if uid in self.neuron_stats:
                        self.neuron_changes[uid][block]['old_stats'] = self.neuron_stats[uid]

                if uid in self.neuron_stats:
                    del self.neuron_stats[uid]
                    changed_hotkeys += [uid]

        if len(changed_hotkeys):
            logger.info(f"Hotkeys changed: {changed_hotkeys}")
            self.save()  # save neuron_stats, neuron_hotkeys, and neuron_changes to filesystem

    def neuron_stats_update(self, neuron_stats: Dict[int, Dict[str, Any]]):
        r""" Updates self.neuron_stats with new individual dictionaries per uid.
        """
        responsive_uids = []
        for _uid, _stats in neuron_stats.items():
            stats = self.neuron_stats.setdefault(_uid, {})

            # === EMA normal update ===
            # If synapse responsive push available values into EMA for normal update.
            # Normal EMA values provide a view on neuron performance if fully responsive.
            for key in _stats:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                if math.isnan(_stats[key]):
                    continue
                if key in stats:
                    stats[key] = (1 - self.alpha) * stats[key] + self.alpha * _stats[key]  # update EMA
                else:
                    stats.setdefault(key, _stats[key])

            # === Extra stats computation ===
            # Compute values on EMA stats, such as the scaling law on EMA loss.
            # Required for values that need to be computed on longer-term stats.
            extra_stats = {}
            if 'loss_nxt' in _stats and 'loss_nxt' in stats:  # elif neuron not responsive then omit
                # estimate the effective number of model parameters from EMA loss
                _num_params = scaling_law_loss_to_params(torch.tensor(stats['loss_nxt']))

                # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
                _pow_num_params = torch.pow(_num_params, self.config.nucleus.scaling_law_power)

                extra_stats.update({'est_params_nxt': _num_params.item(), 'base_params_nxt': _pow_num_params.item()})

                if 'synergy_nxt' in stats:
                    extra_stats['shapley_values_nxt'] = extra_stats['base_params_nxt'] + stats['synergy_nxt']

                if 'logits_excess_nxt' in stats:
                    # penalize by logits divergence excess
                    extra_stats['shapley_values_nxt'] /= 1 + stats['logits_excess_nxt']

            # === EMA zeroing update ===
            # Push zero into EMA for synapse_keys to exponentially decay weighting keys if neuron non-responsive
            if 'updates!' in stats:
                stats['updates!'] += 1  # increment number of EMA zeroing updates
            else:
                stats.setdefault('updates!', 1)  # number of EMA zeroing updates init to zero

            for key in self.synapse_keys:
                zkey = key + '!'  # zeroing key
                stats.setdefault(zkey, 0.)  # initialize zkey val to zero to gradually increase with observations
                if key in _stats and not math.isnan(_stats[key]):
                    responsive_uids += [_uid]
                    stats[zkey] = (1 - self.alpha) * stats[zkey] + self.alpha * _stats[key]
                elif key in extra_stats and not math.isnan(extra_stats[key]):
                    responsive_uids += [_uid]
                    stats[zkey] = (1 - self.alpha) * stats[zkey] + self.alpha * extra_stats[key]
                else:
                    stats[zkey] = (1 - self.alpha) * stats[zkey]  # + self.alpha * 0

            # === EMA normal update ===
            # If synapse responsive push available values into EMA for normal update.
            # Normal EMA values provide a view on neuron performance if fully responsive.
            for key in self.synapse_keys:
                if key in _stats or key in extra_stats:
                    updates = 'updates_' + key
                    if updates in stats:
                        stats[updates] += 1  # increment number of normal EMA updates made
                    else:
                        stats.setdefault(updates, 1)  # add updates fields for new uid entries

            for key in extra_stats:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                if math.isnan(extra_stats[key]):
                    continue
                if key in stats:
                    stats[key] = (1 - self.alpha) * stats[key] + self.alpha * extra_stats[key]  # update EMA
                else:
                    stats.setdefault(key, extra_stats[key])

        return responsive_uids, list(neuron_stats.keys())  # responsive_uids, queried_uids

    def calculate_weights(self):
        r""" Calculates neuron set-weights from weight_key mapped values. Defines weight_key as the neuron stats key
        used to obtain the mapped stat value (typically a Shapley value) that the final set-weights are calculated from.
        """

        weight_key = self.weight_key + '!'  # use zeroing key to penalize non-responsive neurons

        min_allowed_weights = self.subtensor.min_allowed_weights
        max_weight_limit = self.subtensor.max_weight_limit

        # === Populate neuron weights ===
        neuron_weights = torch.zeros_like(self.metagraph.S)  # allow unevaluated UIDs for min_allowed_weights
        for uid in self.neuron_stats:
            if weight_key in self.neuron_stats[uid]:
                neuron_weights[uid] = torch.tensor([self.neuron_stats[uid][weight_key]])

        # === Filter to non-zero weights ===
        sample_uids = torch.argwhere(neuron_weights > 0).squeeze(dim=1)  # find uids with non-zero weight
        sample_weights = neuron_weights[sample_uids]  # filter to non-zero weights

        # === If no uids responds, return ===
        if len(sample_uids) == 0:
            return sample_uids, sample_weights

        # === Exclude lowest quantile from weight setting ===
        max_exclude = (len(sample_weights) - min_allowed_weights) / len(sample_weights)  # max excludable weight quantile
        quantile = self.subtensor.validator_exclude_quantile if self.config.neuron.exclude_quantile == -1 else self.config.neuron.exclude_quantile 
        if 0 < max_exclude:
            exclude_quantile = min([quantile , max_exclude])  # reduce quantile to meet min_allowed_weights
            lowest_quantile = sample_weights.quantile(exclude_quantile)  # find lowest quantile threshold
            sample_uids = sample_uids[lowest_quantile <= sample_weights]  # exclude uids with weights below quantile
            sample_weights = sample_weights[lowest_quantile <= sample_weights]  # exclude weights below quantile

            logger.info(f'Exclude {exclude_quantile} quantile ({lowest_quantile}) | '
                        f'{len(sample_weights)} Shapley values | min:{sample_weights.min()} max:{sample_weights.max()}')

        # === Normalize and apply max_weight_limit ===
        sample_weights = bittensor.utils.weight_utils.normalize_max_weight(x=sample_weights,
                                                                             limit=max_weight_limit)
        logger.info(f'{len(sample_weights)} normalize_max_weight | '
                    f'max:{sample_weights.max()}')

        return sample_uids, sample_weights

    def weights_table(self, sample_uids, sample_weights, include_uids=None, num_rows: int = None):
        r""" Prints weights table given sample_uids and sample_weights.
        """
        min_allowed_weights = self.subtensor.min_allowed_weights
        max_weight_limit = self.subtensor.max_weight_limit

        # === Weight table ===
        # Prints exponential moving average statistics of valid neurons and latest weights
        _neuron_stats = {}
        uid_weights = []  # (uid, weight) tuples for sorting to find top/bottom weights
        unvalidated = []
        for uid, weight in zip(sample_uids.tolist(), sample_weights.tolist()):
            if uid in self.neuron_stats:
                _neuron_stats[uid] = {k: v for k, v in self.neuron_stats[uid].items()}
                _neuron_stats[uid]['weight'] = weight
                uid_weights += [(uid, weight)]
            else:
                unvalidated += [uid]

        if include_uids is not None and num_rows is not None:
            sorted_uids = sorted(uid_weights, key=lambda tup: tup[1])
            top_bottom_uids = [_uid for _uid, _ in sorted_uids[:5] + sorted_uids[-10:]]
            _include_uids = set(include_uids) | set(top_bottom_uids)
            avail_include_uids = list(set(_neuron_stats.keys()) & _include_uids)  # exclude include_uids with no stats
            if len(_neuron_stats) > num_rows:  # limit table to included_uids and remaining sample up to num_rows
                remaining_uids = set(_neuron_stats.keys()) - _include_uids  # find sample remaining, loses sample ordering
                remaining_uids = [uid for uid in _neuron_stats if uid in remaining_uids]  # recover sample ordering
                limited_uids = avail_include_uids + remaining_uids[:num_rows - len(_include_uids)]
                _neuron_stats = {uid: stats for uid, stats in _neuron_stats.items() if uid in limited_uids}

        print()
        stats_table(_neuron_stats, 'weight', self.config.get('width', None),
                    f'[white] Neuron weights [/white] | ' + str(self),  # title
                    f'Validated {min_allowed_weights}/'
                    f'[bold]{len(self.neuron_stats)}[/bold]/{self.metagraph.n} (min/[bold]valid[/bold]/total) | '
                    f'sum:{sample_weights.sum().item():.2g} '
                    f'[white] max:[bold]{sample_weights.max().item():.4g}[/bold] / '
                    f'min:[bold]{sample_weights.min().item():.4g}[/bold] [/white] '
                    f'\[{max_weight_limit:.4g} allowed]',  # caption
                    mark_uids=include_uids)



if __name__ == '__main__':
    neuron().run()


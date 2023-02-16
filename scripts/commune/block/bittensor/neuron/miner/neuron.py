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

""" Template server.

Example:
    $ import neurons
    $ neurons.text.core_server.neuron().run()
"""

import bittensor
import os

from .nucleus import server


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
""" The Exodus base client.

Example:
    $ python miners/text/template_client.py

"""
import bittensor
import sys
import time
import datetime
from threading import Lock
from datetime import datetime,timedelta
from loguru import logger; logger = logger.opt(colors=True)
from torch.nn.utils.rnn import pad_sequence

import wandb
import pandas
# Prometheus
from prometheus_client import Counter, Gauge, Histogram, Summary, Info, CollectorRegistry
# Torch
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

class neuron:
    r"""
    Creates a bittensor neuron that specializes in the serving. The template server miner
    serves a NLP model from huggingface on the bittensor network. By default, the model does 
    not train itself and thus requires less memory to run. 

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            lasthidden (:obj:bool, `optional`):
                lasthidden synapse control
            causallm (:obj:bool, `optional`):
                causallm synapse control
            causallmnext (:obj:bool, `optional`):
                causallmnext synapse control
            seq2seq (:obj:bittensor.metagraph, `optional`):
                seq2seq synapse control
            synapse_list (:obj:list of int, `optional`):
                

    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> server = bittensor.neuron.text.core_server.neuron(subtensor=subtensor)
            >>> server.run()
    """
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        model: 'Model' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,
        lasthidden = None,
        causallm = None,
        causallmnext = None,
        seq2seq = None,
        synapse_list = None,
        relay_enabled = True,
    ):
        if config == None: config = server.config()
        config = config; 
        
        self.relay_enabled = relay_enabled

        if synapse_list != None:
            config.neuron.lasthidden = False
            config.neuron.causallm = False
            config.neuron.causallmnext = False
            config.neuron.seq2seq = False

            if bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE in synapse_list:
                config.neuron.lasthidden = True
            
            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM in synapse_list:
                config.neuron.causallm = True

            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT in synapse_list:
                config.neuron.causallmnext = True

            if bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ in synapse_list:
                config.neuron.seq2seq = True

        config.neuron.lasthidden = lasthidden if lasthidden != None else config.neuron.lasthidden
        config.neuron.causallm = causallm if causallm != None else config.neuron.causallm
        config.neuron.causallmnext = causallmnext if causallmnext is not None else config.neuron.causallmnext
        config.neuron.seq2seq = seq2seq if seq2seq != None else config.neuron.seq2seq

        self.check_config( config )
        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )
        # Init prometheus.
        # By default we pick the prometheus port to be axon.port - 1000 so that we can match port to server.
        bittensor.prometheus ( 
            config = config,
            port = config.prometheus.port if config.axon.port == bittensor.defaults.axon.port else config.axon.port - 1000
        )

        self.model = server(config = config, model=model)
        self.config = config
        self.config.to_prometheus()

        self.subtensor = subtensor
        self.wallet = wallet
        self.axon = axon
        self.metagraph = metagraph

    def run(self):
        self.serve(
            self.config,
            self.model,
            subtensor = self.subtensor,
            wallet = self.wallet,
            axon = self.axon,
            metagraph = self.metagraph,
        )


    @classmethod
    def config(cls):
        return server.config()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.prometheus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name), config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)


    @staticmethod
    def serve( 
            config, 
            model,
            subtensor = None,
            wallet = None,
            axon= None,
            metagraph = None,
        ):
        config.to_defaults()
        model= model.to(model.device)

        # Create Subtensor connection
        subtensor = bittensor.subtensor(config = config) if subtensor == None else subtensor

        # Load/Create our bittensor wallet.
        
        
        if wallet == None:
            wallet = bittensor.wallet( config = config )
            if config.neuron.wait_until_registered:
                time_waiting = 0
                check_registration_period = 60
                while not wallet.is_registered(subtensor=subtensor):
                    time.sleep(check_registration_period)
                    time_waiting += check_registration_period
                    print(f'Not registered: waiting for wallet: {wallet} ({time_waiting}s)')
            else:
                wallet=wallet.create().reregister(subtensor=subtensor) 
        else:
            wallet.reregister(subtensor=subtensor)

        # Load/Sync/Save our metagraph.
        if metagraph == None:
            metagraph = bittensor.metagraph ( 
                subtensor = subtensor
            )
        
        metagraph.load().sync().save()

        # Create our optimizer.
        optimizer = torch.optim.SGD(
            [ {"params": model.parameters()} ],
            lr = config.neuron.learning_rate,
            momentum = config.neuron.momentum,
        )
        mutex = Lock()

        # --- Setup prometheus summaries.
        # These will not be posted if the user passes --prometheus.level OFF
        registry = CollectorRegistry()
        prometheus_counters = Counter('neuron_counters', 'Counter sumamries for the running server-miner.', ['neuron_counters_name'], registry=registry)
        prometheus_guages = Gauge('neuron_guages', 'Guage sumamries for the running server-miner.', ['neuron_guages_name'], registry=registry)
        prometheus_info = Info('neuron_info', "Info sumamries for the running server-miner.", registry=registry)
        prometheus_guages.labels( 'model_size_params' ).set( sum(p.numel() for p in model.parameters()) )
        prometheus_guages.labels( 'model_size_bytes' ).set( sum(p.element_size() * p.nelement() for p in model.parameters()) )
        prometheus_info.info ({
            'type': "core_server",
            'uid': str(metagraph.hotkeys.index( wallet.hotkey.ss58_address )),
            'network': config.subtensor.network,
            'coldkey': str(wallet.coldkeypub.ss58_address),
            'hotkey': str(wallet.hotkey.ss58_address),
        })

        timecheck_dicts = {bittensor.proto.RequestType.FORWARD:{}, bittensor.proto.RequestType.BACKWARD:{}}
        n_topk_peer_weights = subtensor.min_allowed_weights

        def priority(pubkey:str, request_type:bittensor.proto.RequestType, inputs_x) -> float:
            r"""Calculates the priority on requests based on stake and size of input
                Args:
                    pubkey ( str, `required`):
                        The public key of the caller.
                    inputs_x ( :obj:`torch.Tensor`, `required`):
                        torch inputs to be forward processed.
                    request_type ( bittensor.proto.RequestType, `required`):
                        the request type ('FORWARD' or 'BACKWARD').
            """
            try:        
                uid = metagraph.hotkeys.index(pubkey)
                priority = metagraph.S[uid].item()
            
            except:
                # zero priority for those who are not registered.
                priority =  0

            return priority

        def forward_generate( inputs_x:torch.FloatTensor, synapse, model_output = None):
            tokens = model.token_remap(inputs_x.to(model.device))
            output = model.model.generate(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                max_length=max(tokens['input_ids'].shape[1] + 1, synapse.num_to_generate),
                num_beams=synapse.num_beams,
                no_repeat_ngram_size=synapse.no_repeat_ngram_size,
                early_stopping = synapse.early_stopping,
                do_sample=synapse.do_sample,
                top_p=synapse.top_p,
                num_return_sequences=synapse.num_return_sequences,
                temperature = synapse.temperature,
                repetition_penalty = synapse.repetition_penalty,
                length_penalty = synapse.length_penalty,
                max_time = synapse.max_time,
                num_beam_groups = synapse.num_beam_groups,
            )
            raw_texts = [model.tokenizer.decode(out) for out in output]
            tokens = [model.std_tokenizer.encode(raw_text, return_tensors="pt")[:,:synapse.num_to_generate].view(-1) for raw_text in raw_texts]
            bittensor_output = pad_sequence(tokens, batch_first=True)
            return None, model_output, bittensor_output

        def forward_hidden_state(inputs_x:torch.FloatTensor, synapse, model_output = None):
            with mutex:
                message, model_output, hidden = model.encode_forward(inputs_x.to(model.device), model_output=model_output)
            return message, model_output, hidden

        def forward_casual_lm(inputs_x:torch.FloatTensor, synapse, model_output = None):
            with mutex:
                message, model_output, logits = model.encode_forward_causallm(inputs_x.to(model.device), model_output=model_output)
            return message, model_output, logits

        def forward_casual_lm_next(inputs_x: torch.FloatTensor, synapse, model_output=None):
            with mutex:
                message, model_output, topk_token_phrases = model.encode_forward_causallmnext(inputs_x.to(model.device),
                                                                                            topk=synapse.topk,
                                                                                            model_output=model_output)
            # topk_token_phrases: [sum_b(sum_k(len(phrase_k) + 1)_b)] contains topk token phrases and probabilities
            #   Compacted 1-D tensor >= batch_size * (2 * topk + 1)
            return message, model_output, topk_token_phrases

        def optimizer_step():
            optimizer.step()
            optimizer.zero_grad()

        def blacklist(pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
            r"""Axon security blacklisting, used to blacklist message from low stake members
                Args:
                    pubkey ( str, `required`):
                        The public key of the caller.
                    request_type ( bittensor.proto.RequestType, `required`):
                        the request type ('FORWARD' or 'BACKWARD').
            """
            # Check for registrations

            def registration_check():
                # If we allow non-registered requests return False = not blacklisted.
                is_registered = pubkey in metagraph.hotkeys
                if not is_registered:
                    if config.neuron.blacklist_allow_non_registered:
                        return False

                    prometheus_counters.labels("blacklisted.registration").inc()

                    raise Exception('Registration blacklist')

            # Check for stake
            def stake_check() -> bool:
                # Check stake.
                uid = metagraph.hotkeys.index(pubkey)
                if metagraph.S[uid].item() < config.neuron.blacklist.stake:
                    prometheus_counters.labels("blacklisted.stake").inc()

                    raise Exception('Stake blacklist')
                return False

            # Check for time
            def time_check():
                current_time = datetime.now()
                # Only check if the request are forward requests
                timecheck = timecheck_dicts[request_type]
                if pubkey in timecheck.keys():
                    prev_time = timecheck[pubkey]
                    if current_time - prev_time >= timedelta(seconds=config.neuron.blacklist.time):
                        timecheck[pubkey] = current_time
                    else:
                        timecheck[pubkey] = current_time
                        prometheus_counters.labels("blacklisted.time").inc()

                        raise Exception('Time blacklist')
                else:
                    timecheck[pubkey] = current_time
            
                return False

            # Black list or not
            try:
                registration_check()
                time_check()
                stake_check()            
                return False
            except Exception as e:
                prometheus_counters.labels("blacklisted").inc()
                return True
        
        def synapse_check(synapse, hotkey):
            """
                Custom synapse function to protect certain synapse functions depending on the stake and weight.
                Certain synapses require more compute than others. For instance, TEXT_SEQ_2_SEQ requires a significantly
                more commitment by the server than a requeset for TEXT_CAUSAL_LM_NEXT.

                Args:
                    synapse (:obj:`bittensor.proto.SynapseArgs`, `required`): 
                        The proto message that contains additional args for individual synapse functions
                    hotkey (:obj:`torch.FloatTensor`, `required`):
                        The hotkey that sent the request

            """
            ## Uid that sent the request
            incoming_uid = metagraph.hotkeys.index(hotkey)
            if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
                
                if metagraph.S[incoming_uid] < config.neuron.lasthidden_stake:
                    return False
                
            elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:

                if metagraph.S[incoming_uid] < config.neuron.causallm_stake:
                    return False

            elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:

                if metagraph.S[incoming_uid] < config.neuron.causallmnext_stake:
                    return False

            elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:

                if (metagraph.S[incoming_uid] < config.neuron.seq2seq_stake) and (metagraph.S[incoming_uid,  uid]):
                    return False     
            else:
                return False

            return True

        def backward_callback(inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
            """
                The default backward callback when no callback is attached: Is used to call specific synapse functions

                Args:
                    inputs_x (:obj:`torch.FloatTensor`, `required`): 
                        The inputs that will be passed to the synapse functions
                    grads_dy (:obj:`torch.FloatTensor`, `required`):
                        The gradients that will be passed to the synapse functions
                    synapses (:obj: list of bittensor.proto.SynapseArgs, 'Optional')
                        The proto message that contains additional args for individual synapse functions

                Returns:
                    response_tensors: (:obj: list of bittensor.proto.Tensor, `required`): 
                        serialized tensor response from the nucleus call or None.
                    response_codes: (:obj: list of bittensor.proto.ReturnCode, `required`)
                        return code associated with forward call i.e. Success of Timeout.
                    response_messages: (:obj: list of strings, `required`)
                        return message associated with synapse call
            """
            # --- initialize response variables --- 
            response_tensors = []
            response_codes = []
            response_messages = []
            
            if not config.neuron.remote_train:
                return response_tensors, response_codes, response_messages

            # --- calling attached synapses ---
            with mutex and torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
                for index, synapse in enumerate(synapses):
                    try:
                        if synapse.synapse_type in axon.synapse_callbacks and axon.synapse_callbacks[synapse.synapse_type] != None:
                            message, model_output, response_tensor = axon.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                            grads_dy_norm = grads_dy[index]/(grads_dy[index].sum() + 0.00001)
                            torch.autograd.backward (
                                tensors = [ response_tensor ],
                                grad_tensors = [ grads_dy_norm ],
                                retain_graph=True
                            )
                            # Only consider loss from causal LM next.
                            if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:
                                model.remote_losses.append(model_output.loss)
                                model.remote_losses = model.remote_losses[-config.neuron.num_remote_loss:] if len(model.remote_losses) > config.neuron.num_remote_loss else model.remote_losses
                            model.backward_gradients_count += inputs_x[index].size(0)
                            response_tensors.append(None)
                            response_codes.append(bittensor.proto.ReturnCode.Success)
                            response_messages.append('Success')
                            
                        else:
                            response_tensors.append(None)
                            response_codes.append(bittensor.proto.ReturnCode.NotImplemented)
                            response_messages.append('Not Implemented')
                    except Exception as e:
                        # --- Exception Hit in Synapse ---
                        response_tensors.append(None)
                        response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                        response_messages.append(str(e))

            return response_tensors, response_codes, response_messages

        # Create our axon server and subscribe it to the network.
        if axon == None:
            axon = bittensor.axon(
                config = config,
                wallet = wallet,
                synapse_checks=synapse_check,
                synapse_last_hidden = forward_hidden_state if model.config.neuron.lasthidden else None,
                synapse_causal_lm = forward_casual_lm if model.config.neuron.causallm else None,
                synapse_causal_lm_next = forward_casual_lm_next if model.config.neuron.causallmnext else None,
                synapse_seq_2_seq = forward_generate if model.config.neuron.seq2seq else None ,
                blacklist = blacklist if not model.config.neuron.disable_blacklist else None,
                priority = priority if not model.config.neuron.disable_priority else None,
            ).start().serve(subtensor=subtensor)
        
        axon.optimizer_step = optimizer_step
        axon.attach_backward_callback(backward_callback)
        # Training Data
        if config.neuron.local_train:
            dataset = bittensor.dataset(config=config)
            dataset.set_data_size(10, 64)
            data = next(dataset)

        # load our old model
        if not config.neuron.restart :
            model.load(config.neuron.full_path)

        if config.wandb.api_key != 'default':
            # --- Init Wandb.
            bittensor.wandb(
                config = config,
                cold_pubkey = wallet.coldkeypub.ss58_address,
                hot_pubkey = wallet.hotkey.ss58_address,
                root_dir = config.neuron.full_path
            )

        last_set_block = subtensor.get_current_block()
        blocks_per_epoch = subtensor.blocks_per_epoch if config.neuron.blocks_per_epoch == -1 else config.neuron.blocks_per_epoch
        blocks_per_set_weights = subtensor.blocks_per_epoch if config.neuron.blocks_per_set_weights == -1 else config.neuron.blocks_per_set_weights

        # --- Run Forever.
        while True:
            iteration = 0
            local_data = {}
            nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
            uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
            current_block = subtensor.get_current_block()
            end_block = current_block + config.neuron.blocks_per_epoch
            if config.neuron.local_train:
                # --- Training step.
                while end_block >= current_block:
                    if current_block != subtensor.get_current_block() and axon.priority_threadpool.is_empty:
                        with mutex:
                            logger.info(f'local training\titeration: {iteration}\tstart')
                            loss, _ = model( next(dataset).to(model.device) )
                            if iteration > 0 : 
                                losses += loss
                            else:
                                losses = loss
                            iteration += 1
                            current_block = subtensor.get_current_block()
                            logger.info(f'local training\titeration: {iteration}\tloss: {loss}')
                    else:
                        time.sleep(1)
                
                if iteration != 0:
                    (losses/iteration).backward()
            
            else:
                while end_block >= current_block:
                    time.sleep(12)
                    current_block = subtensor.get_current_block()

            # --- Update parameters
            if (config.neuron.local_train and iteration > 0) or (config.neuron.remote_train and model.backward_gradients_count > 0):
                # Custom learning rate
                if model.backward_gradients_count > 0:
                    optimizer.param_groups[0]['lr'] =  0.1/(model.backward_gradients_count)
                else:
                    optimizer.param_groups[0]['lr'] =  0.1

                logger.info('Optmization Started')
                with mutex:
                    clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                logger.info('Optimization Successful: Model updated')

                if (config.neuron.local_train and iteration > 0):
                    local_data = {'local/loss': losses.detach().item() / iteration}

                    if local_data['local/loss'] < model.best_loss:
                        model.best_loss = local_data['local/loss']
                        model.save(config.neuron.full_path)

                # Save it only when it gives a low average loss over a large sample size (config.neuron.num_remote_loss), default to 20. 
                elif (config.neuron.remote_train and len(model.remote_losses) >= config.neuron.num_remote_loss):
                    local_data = {'local/remote_loss': sum(model.remote_losses) / len(model.remote_losses)}

                    if local_data['local/remote_loss'] < model.best_remote_loss:
                        model.best_remote_loss = local_data['local/remote_loss']
                        model.save(config.neuron.full_path)

                    model.remote_losses = []

                model.backward_gradients_count = 0
                
            wandb_data = {            
                'stake': nn.stake,
                'rank': nn.rank,
                'trust': nn.trust,
                'consensus': nn.consensus,
                'incentive': nn.incentive,
                'emission': nn.emission,
            }
            
            if config.wandb.api_key != 'default':

                df = pandas.concat( [
                    bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                    axon.to_dataframe( metagraph = metagraph ),
                ], axis = 1)
                df['uid'] = df.index
                wandb_info_axon = axon.to_wandb()                
                wandb.log( { **wandb_data, **wandb_info_axon, **local_data }, step = current_block )
                wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

            # === Prometheus logging.
            prometheus_guages.labels("stake").set( nn.stake )
            prometheus_guages.labels("rank").set( nn.rank )
            prometheus_guages.labels("trust").set( nn.trust )
            prometheus_guages.labels("consensus").set( nn.consensus )
            prometheus_guages.labels("incentive").set( nn.incentive )
            prometheus_guages.labels("emission").set( nn.emission )

            if current_block - last_set_block > blocks_per_set_weights:
                bittensor.__console__.print('[green]Current Status:[/green]', {**wandb_data, **local_data})
                metagraph.sync()
                last_set_block = current_block
                if not config.neuron.no_set_weights and False:
                    try: 
                        bittensor.__console__.print('[green]Current Status:[/green]', {**wandb_data, **local_data})
                        # Set self weights to maintain activity.
                        # --- query the chain for the most current number of peers on the network
                        chain_weights = torch.zeros(subtensor.n)
                        chain_weights [ uid ] = 1 
                        did_set = subtensor.set_weights(
                            uids=torch.arange(0,subtensor.n),
                            weights = chain_weights,
                            wait_for_inclusion = False,
                            wallet = wallet,
                        )
                        if did_set:
                            logger.success('Successfully set weights on the chain')
                        else:
                            logger.error('Failed to set weights on chain. (Timeout)')
                        
                    except Exception as e:
                        logger.error('Failure setting weights on chain with error: {}', e)



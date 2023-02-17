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



neuron_stats_columns = [
    ['UID', 'uid', '{:.0f}', 'cyan'],  # neuron UID
    ['Upd!', 'updates!', '{}', 'bright_yellow'],  # number of exponential moving average updates with zeroing on
    ['nUpd', 'updates_shapley_values_nxt', '{}', 'bright_yellow'],  # number of exponential moving average updates to nShap
    ['mUpd', 'updates_shapley_values_min', '{}', 'bright_yellow'],  # number of exponential moving average updates to mShap
    ['nTime', 'response_time_nxt', '{:.2f}', 'yellow'],  # response time to TextCausalLMNext forward requests [TextCausalLMNext]
    ['sTime', 'response_time', '{:.2f}', 'yellow'],  # response time to TextCausalLM forward requests
    ['Route', 'routing_score', '{:.3f}', 'grey30'],  # validator routing score (higher preferred)
    ['Weight', 'weight', '{:.5f}', 'green'],  # weight set on substrate (each epoch)
    ['nShap!', 'shapley_values_nxt!', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for phrase validation (zeroing) [TextCausalLMNext]
    ['nShap', 'shapley_values_nxt', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for phrase validation [TextCausalLMNext]
    ['mShap!', 'shapley_values_min!', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley (zeroing)
    ['mShap', 'shapley_values_min', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley
    ['sLoss', 'loss', '{:.2f}', 'bright_cyan'],  # next token prediction loss average over sequence
    ['vLoss', 'loss_val', '{:.2f}', 'bright_cyan'],  # next token prediction loss for validation task
    ['nvLoss', 'loss_val_nxt', '{:.2f}', 'bright_cyan'],  # next token prediction loss for validation task [TextCausalLMNext]
    ['nLoss', 'loss_nxt', '{:.2f}', 'bright_cyan'],  # next token phrase prediction loss for phrase validation task [TextCausalLMNext]
    ['RLoss', 'routing_loss', '{:.3f}', 'grey30'],  # MSE between routing_score and conditioned loss
    ['nRLoss', 'routing_loss_nxt', '{:.3f}', 'grey30'],  # MSE between routing_score_nxt and conditioned loss [TextCausalLMNext]
    ['sShap', 'shapley_values', '{:.0f}', 'magenta'],  # Shapley value (=Base+Syn) over sequence
    ['vShap', 'shapley_values_val', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for validation
    ['sBase', 'base_params', '{:.0f}', ''],  # parameter count estimate via adjusted scaling law
    ['vBase', 'base_params_val', '{:.0f}', ''],  # square root parameter count estimate for validation task
    ['nBase', 'base_params_nxt', '{:.0f}', ''],  # square root parameter count estimate for phrase validation task [TextCausalLMNext]
    ['nParam~', 'est_params_nxt', '{:.2g}', 'magenta'],  # parameter count estimate for phrase validation task [TextCausalLMNext]
    ['nDiv', 'logits_divergence_nxt', '{:.2g}', ''],  # logits divergence avg compared to network prob dist [TextCausalLMNext]
    ['nExc', 'logits_excess_nxt', '{:.2f}', ''],  # logits divergence excess avg above network avg + std [TextCausalLMNext]
    ['sSyn', 'synergy', '{:.0f}', 'white'],  # Shapley pairwise synergy over sequence loss (parameter count estimate)
    ['vSyn', 'synergy_val', '{:.0f}', 'white'],  # Shapley pairwise synergy over validation loss (count estimate)
    ['nSyn', 'synergy_nxt', '{:.0f}', 'white'],  # Shapley pairwise synergy over phrase validation loss (count estimate) [TextCausalLMNext]
    ['sSynD', 'synergy_loss_diff', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over sequence loss (loss difference)
    ['vSynD', 'synergy_loss_diff_val', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over validation loss (loss difference)
    ['nSynD', 'synergy_loss_diff_nxt', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over phrase validation loss (loss difference) [TextCausalLMNext]
]


def scaling_law_loss_to_params(loss):
    r""" (OpenAI scaling laws) Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv:2001.08361 (2020)
    """
    num_params = torch.exp(torch.log(torch.tensor(8.8e13).to(loss.device)) -
                           torch.log(torch.clamp(loss, 1.69)) / 0.076)  # loss lower bound 1.69 is entropy of natural text
    return num_params


def textcausallm(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                 times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                 inputs: torch.FloatTensor, validation_len: int, loss_fct: Callable,
                 scaling_law_power: float, synergy_scaling_law_power: float,
                 console_width: int, logging, synapse: 'bittensor.TextCausalLM' = None, index_s: int = 0
                 ) -> Tuple[torch.FloatTensor, Dict]:
    r"""
    Calculate Shapley values and neuron response validation measure statistics, given TextCausalLM synapse responses.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            inputs (:obj:`torch.FloatTensor`, `required`):
                [batch_size, sequence_len + validation_len] Token batch of original inputs with validation tokens.
            validation_len (:obj:`int`, `required`):
                Number of held-out phrase token batch for extended validation, not sent to neurons.
            loss_fct (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            scaling_law_power (:obj:`float`, `required`):
                Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            synergy_scaling_law_power (:obj:`float`, `required`):
                Power for synergy modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            console_width (:obj:`int`, `required`):
                Config console width for table print.
            logging (:obj:`bool`, `required`):
                Log tables to console.
            synapse (:obj:`bittensor.TextCausalLM`, `optional`):
                TextCausalLM synapse object.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
    """

    inputs_seq = inputs[..., :-validation_len]  # input sequence without last token [batch_size, sequence_len]
    inputs_val = inputs[..., -validation_len]  # input validation with next token [batch_size]

    def _base_params(_stats, query_response):
        _stats.update({'logits': query_response[:, :-1, :],
                       'logits_val': query_response[:, -1:, :]})

        for target, _ext in [(inputs_seq[:, 1:], ''), (inputs_val, '_val')]:
            _loss = calc_loss_fct(loss_fct, _stats['logits' + _ext], target)  # CausalLM loss
            if _loss.isnan() or _loss.isinf():
                _loss = 20  # assign large loss

            # estimate the effective number of model parameters, modified with the scaling_law_power
            _num_params = scaling_law_loss_to_params(_loss)

            # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
            _pow_num_params = torch.pow(_num_params, scaling_law_power)

            _stats.update({'loss' + _ext: _loss,
                           'est_params' + _ext: _num_params, 'base_params' + _ext: _pow_num_params,
                           'synergy' + _ext: 0, 'synergy_loss_diff' + _ext: 0})

    def _synergy(first, second, target, _ext):
        # Combined logits: log of average probabilities per token between responses
        combined_logits = torch.log((torch.softmax(first['logits' + _ext], dim=-1) +
                                     torch.softmax(second['logits' + _ext], dim=-1)) / 2 + 1e-40)
        measured_loss = calc_loss_fct(loss_fct, combined_logits, target)  # actual measured loss

        return measured_loss

    shapley_start_time = time.time()

    loss, stats, unsuccessful = shapley_base(uids, query_responses, return_ops, times, routing_score,
                                             _base_params, index_s, ext='')

    logger.info(f'{str(synapse)} \t| Shapley base values (power={scaling_law_power:.1f})'
                f'<dim>[{time.time() - shapley_start_time:.3g}s]</dim>')

    synergy_start_time = time.time()

    syn_loss_diff = shapley_synergy(stats, _synergy, ext='', target=inputs_seq[:, 1:],
                                    scaling_law_power=synergy_scaling_law_power)
    syn_loss_diff_val = shapley_synergy(stats, _synergy, ext='_val', target=inputs_val,
                                        scaling_law_power=synergy_scaling_law_power)

    # === Shapley value combination ===
    # Combine base values with synergy approximation to get final Shapley values.
    for s in stats.values():
        for ext in ['', '_val']:
            if 'base_params' + ext in s and 'synergy' + ext in s:
                s['shapley_values' + ext] = (s['base_params' + ext] + s['synergy' + ext])

            if 'logits' + ext in s:
                del s['logits' + ext]  # remove logits - not needed for stats anymore

        if 'shapley_values' in s and 'shapley_values_val' in s:
            s['shapley_values_min'] = torch.min(s['shapley_values'], s['shapley_values_val'])

        for key in s:
            if hasattr(s[key], 'item'):
                s[key] = s[key].item()

    logger.info(f'{str(synapse)} \t| Shapley synergy values (power={synergy_scaling_law_power:.1f})'
                f'<dim>[{time.time() - synergy_start_time:.3g}s]</dim>')

    if logging:
        # === Synergy table ===
        # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
        synergy_table(stats, syn_loss_diff, 'shapley_values_min', console_width=console_width)

        # === Neuron responses (table) ===
        # Prints the evaluation of the neuron responses to the validator request
        synapse_table(str(synapse), stats, 'shapley_values_min', console_width, shapley_start_time)

    # === Unsuccessful responses ===
    # Prints the return codes and response times of unsuccessful responses
    unsuccess(str(synapse), unsuccessful)

    return loss, stats


def textcausallmnext(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                     times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                     inputs: torch.FloatTensor, validation_len: int, loss_fct: Callable,
                     scaling_law_power: float, synergy_scaling_law_power: float,
                     console_width: int, logging, synapse: 'bittensor.TextCausalLMNext' = None, index_s: int = 0
                     ) -> Tuple[torch.FloatTensor, Dict]:
    r"""
    Calculate Shapley values and neuron response validation measure statistics, given TextCausalLMNext synapse responses.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            inputs (:obj:`torch.FloatTensor`, `required`):
                [batch_size, sequence_len + validation_len] Token batch of original inputs with validation tokens.
            validation_len (:obj:`int`, `required`):
                Number of held-out phrase token batch for extended validation, not sent to neurons.
            loss_fct (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            scaling_law_power (:obj:`float`, `required`):
                Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            synergy_scaling_law_power (:obj:`float`, `required`):
                Power for synergy modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            console_width (:obj:`int`, `required`):
                Config console width for table print.
            logging (:obj:`bool`, `required`):
                Log tables to console.
            synapse (:obj:`bittensor.TextCausalLMNext`, `optional`):
                TextCausalLMNext Synapse object.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
    """

    inputs_nxt = inputs[..., -validation_len:]  # input validation with next token target phrase [batch_size, val_len]

    def _base_params(_stats, query_response):
        # topk_tensor = unravel_topk_token_phrases(query_response, topk=synapse.topk)  # [batch_size, topk + 1, max_len]
        _losses_val, _losses = phrase_cross_entropy(inputs_nxt, query_response, reduce=False)
        _losses_val[_losses_val.isnan()] = 20  # assign large loss
        _losses[_losses.isnan()] = 20  # assign large loss
        _loss_val = _losses_val.mean()
        _loss = _losses.mean()

        _stats.update({'loss_val_nxt': _loss_val, 'losses_nxt': _losses, 'loss_nxt': _loss,
                       'synergy_nxt': 0, 'synergy_loss_diff_nxt': 0})

    def _synergy(first, second, target, ext):
        # average first + second probabilities per batch item, convert to loss
        measured_loss = -torch.log((torch.exp(-first['losses_nxt']) +
                                    torch.exp(-second['losses_nxt'])) / 2 + 1e-40).mean()

        return measured_loss

    shapley_start_time = time.time()
    loss, stats, unsuccessful = shapley_base(uids, query_responses, return_ops, times, routing_score,
                                             _base_params, index_s, ext='_nxt')
    logger.info(f'{str(synapse)} \t| Shapley base values (power={scaling_law_power:.1f})'
                f'<dim>[{time.time() - shapley_start_time:.3g}s]</dim>')

    divergence_start_time = time.time()
    with torch.no_grad():
        logits_divergence(stats, uids, query_responses, return_ops, times, index_s, ext='_nxt')
    logger.info(f'{str(synapse)} \t| Logits divergences <dim>[{time.time() - divergence_start_time:.3g}s]</dim>')

    synergy_start_time = time.time()
    syn_loss_diff = shapley_synergy(stats, _synergy, '_nxt', scaling_law_power=synergy_scaling_law_power)
    logger.info(f'{str(synapse)} \t| Shapley synergy values (power={synergy_scaling_law_power:.1f})'
                f'<dim>[{time.time() - synergy_start_time:.3g}s]</dim>')

    # === Shapley value combination ===
    # Combine base values with synergy approximation to get final Shapley values.
    for s in stats.values():
        if 'losses_nxt' in s:
            del s['losses_nxt']  # remove batch losses - not needed for stats anymore

        for key in s:
            if hasattr(s[key], 'item'):
                s[key] = s[key].item()

    if logging:
        # === Response table ===
        # Prints the query response table: top prediction probabilities and texts for batch tasks
        batch_predictions = format_predictions(uids, query_responses, return_ops, inputs, validation_len, index_s)
        response_table(batch_predictions, stats, sort_col='loss_nxt', console_width=console_width)

        # === Synergy table ===
        # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
        synergy_table(stats, syn_loss_diff, 'loss_nxt', console_width)

        # === Neuron responses (table) ===
        # Prints the evaluation of the neuron responses to the validator request
        synapse_table(str(synapse), stats, 'loss_nxt', console_width, shapley_start_time)

    # === Unsuccessful responses ===
    # Prints the return codes and response times of unsuccessful responses
    unsuccess(str(synapse), unsuccessful)

    return loss, stats


def shapley_base(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                 times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                 base_params: Callable, index_s: int = 0, ext: str = None) -> Tuple[Union[float, torch.FloatTensor],
                                                                                    Dict,
                                                                                    List]:
    r"""
    Calculate Shapley base values and neuron response validation measure statistics, given responses from a synapse.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            base_params (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            unsuccessful (:obj:`List`, `required`):
                Unsuccessful endpoints [(uid, return_op, time)].
    """
    stats = {}
    unsuccessful = []
    neuron_loss = 0.  # neuron losses to accumulate to then backward() via dendrite
    routing_loss = 0.  # validator routing loss for local model update

    # === Base parameter estimation ===
    # Shapley values - base level - coalition size 1
    # Collect successful neuron responses, calculate base Shapley values.
    # Measured in effective number of model parameters, according to OpenAI scaling laws.
    for index, _uid in enumerate(uids.tolist()):
        if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
            _stats = {'uid': _uid,
                      'response_time' + ext: times[index][index_s],
                      'routing_score': routing_score[_uid]}

            try:
                base_params(_stats, query_responses[index][index_s])

                neuron_loss += _stats['loss' + ext]  # add sequence loss to be backward() to neuron

                # === Add routing loss ===
                # MSE loss between predicted routing score and ideal target routing score.
                # The Bayes risk approx. 1.69, i.e. the minimal loss achievable for next-token
                # prediction on the full distributionP 𝑃, a.k.a the "entropy of natural text"
                # Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." arXiv:2203.15556 (2022).
                routing_score_target = torch.exp(-torch.clamp(_stats['loss' + ext].detach() - 1.69, 0))
                _routing_loss = (routing_score[_uid] - routing_score_target) ** 2  # MSE loss
                routing_loss += _routing_loss
                _stats.update({'routing_score_target' + ext: routing_score_target, 'routing_loss' + ext: _routing_loss})

                stats[_uid] = _stats
            except Exception as e:
                logger.warning(f'Synapse {index_s} error (shapley_base)\t| '
                               f'UID {_uid} <dim>[{times[index][index_s]:.2f}s]</dim>: {e}')
                stats[_uid] = _stats
                unsuccessful += [(_uid, return_ops[index][index_s], times[index][index_s])]
        else:
            stats[_uid] = {'uid': _uid,
                           'response_time' + ext: times[index][index_s],
                           'routing_score': routing_score[_uid]}
            unsuccessful += [(_uid, return_ops[index][index_s], times[index][index_s])]

    return neuron_loss + routing_loss, stats, unsuccessful


def logits_divergence(stats: Dict, uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]],
                      return_ops: List[torch.LongTensor], times: List[torch.FloatTensor],
                      index_s: int = 0, ext: str = None):
    r"""
    Calculate each logits divergence per neuron per task from the average logits over all neurons per task,
    given responses from a synapse.
        Args:
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size.
                Non-responses are zeroes of relevant synapse shape.
                Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.
    """
    probs_k = 0
    probs_avg = None

    # === Probs averaging ===
    # Calculate the average token distribution for each batch task.
    for index, _uid in enumerate(uids.tolist()):
        if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
            try:
                probs = topk_tokens_to_vocab_size(query_responses[index][index_s],
                                                  bittensor.__vocab_size__)  # [batch_size, vocab_size]
                if probs_avg is None:
                    probs_avg = probs
                else:
                    probs_avg += probs
                probs_k += 1

            except Exception as e:
                logger.warning(f'Synapse {index_s} error (logits_divergence)\t| '
                               f'UID {_uid} <dim>[{times[index][index_s]:.2f}s]</dim>: {e}')

    if probs_avg is not None:
        probs_avg /= probs_k
        probs_avg_sqrt = probs_avg.sqrt()
        batch_divergences = []

        # === Distribution divergence ===
        # Calculate the Hellinger distance (f-divergence) from the average probability distribution for each batch task.
        for index, _uid in enumerate(uids.tolist()):
            if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
                try:
                    probs = topk_tokens_to_vocab_size(query_responses[index][index_s],
                                                      bittensor.__vocab_size__)  # [batch_size, vocab_size]
                    divergences = 0.5 * torch.pow(probs.sqrt() - probs_avg_sqrt, 2).sum(dim=1)  # [batch_size] in [0, 1]
                    divergences = divergences.sqrt()
                    stats[_uid]['logits_divergences' + ext] = divergences  # [batch_size]
                    stats[_uid]['logits_divergence' + ext] = divergences.mean()  # scalar
                    batch_divergences += [divergences]

                except Exception as e:
                    logger.warning(f'Synapse {index_s} error (logits_divergence)\t| '
                                   f'UID {_uid} <dim>[{times[index][index_s]:.2f}s]</dim>: {e}')

        batch_divergences = torch.stack(batch_divergences)  # [uids_len, batch_size]
        avg = batch_divergences.mean(dim=0)  # [batch_size]
        std = batch_divergences.std(dim=0)  # [batch_size]

        # logger.info(f"Logits divergences: "
        #             f"avg={', '.join([f'{i}:{v:.3g}' for i, v in enumerate(avg)])}")
        # logger.info(f"std={', '.join([f'{i}:{v:.3g}' for i, v in enumerate(std)])}")

        # === Calculate divergence excess ===
        # For each batch task, calculate excess deviation above a single stddev, in terms of stddev,
        # and apply power to increase score above two stddev, and decrease between one and two stddev.
        # This will effectively allow zero excess below one stddev, and minimal excess below two stddev,
        # but amplify any excess above two stddev (only 2.1% of population for normal dist).
        for _uid, _stats in stats.items():
            if 'logits_divergences' + ext in _stats:
                try:
                    excess = torch.clamp(_stats['logits_divergences' + ext] - (avg + std), 0)  # divergence > avg + std
                    excess /= std + 1e-9  # stddev multiples above 1 stddev
                    excess = torch.pow(excess, 3)  # reduce < 2std, increase > 2std
                    excess = torch.clamp(excess, 0, 10)  # maximum excess ratio of 10

                    _stats['logits_excess' + ext] = excess.mean()  # in [0, 10]
                    del _stats['logits_divergences' + ext]  # keep only scalar stats beyond this

                    # logger.info(f"UID{uid} divergences [{_stats['logits_divergences' + ext].mean():.4g}]: "
                    #             f"{', '.join([f'{i}:{dist:.3g}' for i, dist in enumerate(_stats['logits_divergences' + ext])])}")
                    # logger.info(f"UID{uid} excess [{excess.mean():.3g}]: "
                    #             f"{', '.join([f'{i}:{exc:.3g}' for i, exc in enumerate(excess)])}")

                except Exception as e:
                    logger.warning(f'Synapse {index_s} error (logits_divergence)\t| UID {_uid}: {e}')

def shapley_synergy(stats: Dict, synergy: Callable, ext: str, target: torch.Tensor = None, scaling_law_power: float = 0.5):
    r"""
    Calculates Shapley synergy for coalition size 2, measured performance above expected performance.
    Measured in effective number of model parameters, just like base Shapley values.
        Args:
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            synergy (:obj:`Callable`, `required`)
                Function to calculate measured loss.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.
            target (:obj:`torch.Tensor`, `optional`):
                Target to measure loss against.
            scaling_law_power (:obj:`float`, `optional`):
                Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.

        Returns:
            syn_loss_diff (:obj:`Dict`, `required`):
                Dictionary table of pairwise synergies as loss reductions, with direct loss on diagonal.
    """
    # === Shapley synergy approximation ===
    # Shapley values - second level - coalition size 2
    # Synergy = measured performance above expected performance
    # Measured in effective number of model parameters, just like base Shapley values.
    syn_loss_diff = {}  # expected_loss - measured_loss (where > 0)
    responsives = [uid for uid, stat in stats.items() if 'loss' + ext in stat]
    for _first, first in stats.items():
        if 'loss' + ext not in first:
            continue
        first_diff = syn_loss_diff.setdefault(_first, {})
        first_diff[_first] = first['loss' + ext]  # diagonal keeps direct loss

        for _second, second in stats.items():
            if 'loss' + ext not in second or _second <= _first:
                continue
            second_diff = syn_loss_diff.setdefault(_second, {})

            with torch.no_grad():
                expected_loss = torch.min(first['loss' + ext], second['loss' + ext])  # expecting min loss

                measured_loss = synergy(first, second, target, ext)

                loss_diff_share = torch.clamp(expected_loss - measured_loss, 0) / 2  # record direct loss diff
                loss_diff_share /= len(responsives)  # average over responsives
                first['synergy_loss_diff' + ext] += loss_diff_share
                second['synergy_loss_diff' + ext] += loss_diff_share

                # pairwise loss reduction of expected to measured loss due to synergy between first and second
                first_diff[_second] = loss_diff_share
                second_diff[_first] = loss_diff_share

                measured_params = scaling_law_loss_to_params(measured_loss)
                expected_params = scaling_law_loss_to_params(expected_loss)

                # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
                pow_measured_params = torch.pow(measured_params, scaling_law_power)
                pow_expected_params = torch.pow(expected_params, scaling_law_power)

                synergy_share = torch.clamp(pow_measured_params - pow_expected_params, 0) / 2
                synergy_share /= len(responsives)  # average over responsives
                first['synergy' + ext] += synergy_share  # share synergy amongst coalition members
                second['synergy' + ext] += synergy_share

    return syn_loss_diff


def format_predictions(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]],
                       return_ops: List[torch.LongTensor], inputs: torch.FloatTensor,
                       validation_len: int, index_s: int = 0, number_of_predictions: int = 3) -> List:
    r""" Format batch task topk predictions for rich table print of query responses.
    """
    batch_predictions = []
    std_tokenizer = bittensor.tokenizer()

    # === Batch iteration ===
    for batch_item in range(inputs.shape[0]):
        # === Task formatting ===
        context = inputs[batch_item][:-validation_len]
        answer = inputs[batch_item][-validation_len:]

        context = repr(std_tokenizer.decode(context))[1:-1][-30:]  # strip '' and truncate
        answer = repr(std_tokenizer.decode(answer))[1:-1][:15]  # strip '' and truncate

        task = f"[reverse]{context}[/reverse][bold]{answer}[/bold]"

        # === Prediction formatting ===
        predictions = {}
        for index, uid in enumerate(uids.tolist()):
            if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
                topk_tensor = query_responses[index][index_s]  # [batch_size, (topk + 1), max_len] (prob_k) + floor_prob
                topk_tokens = topk_tensor[batch_item, :-1, 1:].int()  # [batch_size, topk, max_len - 1] Phrase tokens with ignore_index token for padding.
                topk_probs = topk_tensor[batch_item, :-1, 0]  # [batch_size, topk] Probabilities for each phrase in topk

                # === Topk iteration ===
                topk_predictions = ''
                for i in range(number_of_predictions):
                    phrase = topk_tokens[i]
                    phrase = phrase[phrase >= 0]  # strip negative ignore_index = -100
                    phrase_str = repr(std_tokenizer.decode(phrase))[:15]  # decode, escape and truncate
                    prob = f'{topk_probs[i]:.3f}'.lstrip('0').replace('1.000', '1.00')
                    topk_predictions += f"[green]{prob}[/green]: {phrase_str} "

                predictions[uid] = topk_predictions[:-1]  # strip trailing space

        batch_predictions += [(task, predictions)]

    return batch_predictions


def response_table(batch_predictions: List, stats: Dict, sort_col: str, console_width: int,
                   task_repeat: int = 4, tasks_per_server: int = 3):
    r""" Prints the query response table: top prediction probabilities and texts for batch tasks.
    """
    # === Batch permutation ===
    batch_size = len(batch_predictions)
    if batch_size == 0:
        return
    batch_perm = torch.randperm(batch_size)  # avoid restricting observation to predictable subsets

    # === Column selection ===
    columns = [c[:] for c in neuron_stats_columns if c[1] in ['uid', sort_col, 'loss_nxt', 'synergy_nxt', 'logits_excess_nxt']]
    col_keys = [c[1] for c in columns]

    # === Sort rows ===
    sort = sorted([(uid, s[sort_col]) for uid, s in stats.items() if sort_col in s],
                  reverse='loss' not in sort_col, key=lambda _row: _row[1])
    if sort_col in col_keys:
        sort_idx = col_keys.index(sort_col)  # sort column with key of sort_col
        columns[sort_idx][0] += '\u2193'  # ↓ downwards arrow (sort)

    for i, (uid, val) in enumerate(sort):
        # === New table section ===
        if i % task_repeat == 0:
            table = Table(width=console_width, box=None)
            if i == 0:
                table.title = f"[white bold] Query responses [/white bold] | " \
                              f"[white]context[/white][bold]continuation[/bold] | .prob: 'prediction'"

            for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
                table.add_column(col, style=stl, justify='right')

        # === Last table section ===
        if i == len(sort) - 1:
            table.caption = f'[bold]{len(sort)}[/bold]/{len(stats)} (respond/topk) | ' \
                            f'[bold]{tasks_per_server}[/bold] tasks per server | ' \
                            f'repeat tasks over [bold]{task_repeat}[/bold] servers ' \
                            f'[white]\[{math.ceil(1. * len(sort) / task_repeat) * tasks_per_server}/' \
                            f'{batch_size} batch tasks][/white]'

        # === Row addition ===
        row = [txt.format(stats[uid][key]) for _, key, txt, _ in columns]
        for j in range(tasks_per_server):
            batch_item = ((i // task_repeat) * tasks_per_server + j) % batch_size  # repeat task over servers, do not exceed batch_size
            task, predictions = batch_predictions[batch_perm[batch_item]]
            row += [predictions[uid]]

            if i % task_repeat == 0:
                table.add_column(task, header_style='not bold', style='', justify='left')

        table.add_row(*row)

        # === Table print ===
        if (i == len(sort) - 1) or (i % task_repeat == task_repeat - 1):
            try:
                print(table)
            except MarkupError as e:
                print(e)
            else:
                if i == len(sort) - 1:
                    print()


def synergy_table(stats, syn_loss_diff, sort_col, console_width):
    r""" Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
    """
    sort = sorted([(uid, s[sort_col]) for uid, s in stats.items() if sort_col in s],
                  reverse='loss' not in sort_col, key=lambda _row: _row[1])
    uid_col = neuron_stats_columns[0]  # [Column_name, key_name, format_string, rich_style]
    columns = [uid_col] + [[f'{s[0]}', '', '{:.2f}', ''] for s in sort]
    rows = [[uid_col[2].format(s[0])] +
            [('[bright_cyan]{:.2f}[/bright_cyan]' if t == s else
              '[magenta]{:.3f}[/magenta]' if syn_loss_diff[s[0]][t[0]] > 0 else
              '[dim]{:.0f}[/dim]').format(syn_loss_diff[s[0]][t[0]]).replace('0.', '.') for t in sort] for s in sort]

    # === Synergy table ===
    table = Table(width=console_width, box=None)
    table.title = f'[white] Synergy table [/white] | Pairwise synergy'
    table.caption = f'loss decrease'

    for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
        table.add_column(col, style=stl, justify='right')
    for row in rows:
        table.add_row(*row)

    if len(rows):
        print(table)
        print()


def stats_table(stats, sort_col, console_width, title, caption, mark_uids=None):
    r""" Gathers data and constructs neuron statistics table and prints it
    """
    # === Gather columns and rows ===
    if mark_uids is None:
        mark_uids = list()
    stats_keys = [set(k for k in stat)
                  for stat in stats.values() if sort_col in stat]  # all available stats keys with sort_col

    if len(stats_keys) == 0:
        return  # nothing to print

    stats_keys = set.union(*stats_keys)
    columns = [c[:] for c in neuron_stats_columns if c[1] in stats_keys]  # available columns intersecting with stats_keys
    rows = [[('', 0) if key not in stat
             else (('* ' if key == 'uid' and mark_uids and uid in mark_uids else '') + txt.format(stat[key]), stat[key])
             for _, key, txt, _ in columns]
            for uid, stat in stats.items() if sort_col in stat]  # only keep rows with at least one non-empty cell

    if len(columns) == 0 or len(rows) == 0:
        return  # nothing to print

    # === Sort rows ===
    col_keys = [c[1] for c in columns]
    if sort_col in col_keys:
        sort_idx = col_keys.index(sort_col)  # sort column with key of sort_col
        columns[sort_idx][0] += '\u2193'  # ↓ downwards arrow (sort)
        rows = sorted(rows, reverse='loss' not in sort_col, key=lambda _row: _row[sort_idx][1])  # sort according to sortcol

    # === Instantiate stats table ===
    table = Table(width=console_width, box=None, row_styles=[Style(bgcolor='grey15'), ""])
    table.title = title
    table.caption = caption

    for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
        table.add_column(col, style=stl, justify='right')
    for row in rows:
        table.add_row(*[txt for txt, val in row])

    # === Print table ===
    print(table)


def synapse_table(name, stats, sort_col, console_width, start_time):
    r""" Prints the evaluation of the neuron responses to the validator request
    """
    stats_table(stats, sort_col, console_width,
                f'[white] \[{name}] responses [/white] | Validator forward',  # title
                f'[bold]{len([s for s in stats.values() if len(s) and sort_col in s])}[/bold]/'
                f'{len(stats)} (respond/topk) | '
                f'[bold]Synapse[/bold] | [white]\[{time.time() - start_time:.3g}s][/white]'  # caption
                )


def unsuccess(_name, _unsuccessful):
    r""" Prints the return codes and response times of unsuccessful responses
    """
    # === Unsuccessful responses ===
    unsuccess_txt = f'{_name} \t| Unsuccessful <cyan>UID</cyan>[<red>return_op</red> <yellow>time</yellow>]: '
    for _uid, _return_op, _time in _unsuccessful:
        unsuccess_txt += f'{_uid}[<red>{_return_op}</red> <yellow>{_time:.2f}</yellow>] '
    logger.info(unsuccess_txt)



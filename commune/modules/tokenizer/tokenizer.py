
import commune
import torch
from typing import Dict, Any, List, Tuple
import streamlit as st
commune.new_event_loop()
from commune.utils.tokenizer import get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases, \
        encode_topk, decode_topk, prep_tokenizer, check_tokenizer_equivalence

class Tokenizer(c.Module):
    def __init__(self, tokenizer = 'llama', **kwargs):
        self.set_tokenizer(tokenizer=tokenizer, **kwargs)
            
    def set_tokenizer(self, tokenizer='gpt2', **kwargs):
        from transformers import AutoTokenizer, AutoModel
        from commune.utils.tokenizer import prep_tokenizer
        tokenizer = self.shortcuts.get(tokenizer, tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        return {'success': True, 'msg': f'set tokenizer to {tokenizer}'}

    def tokenize(self, text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=64,
                 return_tensors='pt',
                 add_special_tokens=False,
                 **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        sample = self.tokenizer(text, 
                                             padding=padding, 
                                             truncation=truncation, 
                                             max_length=max_length, 
                                             return_tensors=return_tensors,
                                             add_special_tokens=add_special_tokens, 
                                             **kwargs)  # assume tokenizer.padding_side = 'left'
        
        return sample



    def detokenize(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        text = self.tokenizer.batch_decode(input_ids,**kwargs)  # assume tokenizer.padding_side = 'left'

        return text
    
    
    @classmethod
    def test(cls,**kwargs):
        text = 'Whadup'
        self  = cls(**kwargs)
        print(self.tokenize(text))
        print(self.detokenize(self.tokenize(text)['input_ids']))



    shortcuts =  {
        # 0-1B models
        'gpt125m': 'EleutherAI/gpt-neo-125m',

        # 1-3B models
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt3b': 'EleutherAI/gpt-neo-2.7B',
        'opt1.3b': 'facebook/opt-1.3b',
        'opt2.7b': 'facebook/opt-2.7b',

        # 0-7B models
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptjt_mod': 'togethercomputer/GPT-JT-Moderation-6B',
        'gptj': 'EleutherAI/gpt-j-6b',
        'gptj.pyg6b': 'PygmalionAI/pygmalion-6b',
        'gpt6b': 'cerebras/Cerebras-GPT-6.7B',
        'gptj.instruct': 'nlpcloud/instruct-gpt-j-fp16',
        'gptj.codegen': 'moyix/codegen-2B-mono-gptj',
        'gptj.hivemind': 'hivemind/gpt-j-6B-8bit',
        'gptj.adventure': 'KoboldAI/GPT-J-6B-Adventure',
        'gptj.pygppo': 'TehVenom/GPT-J-Pyg_PPO-6B', 
        'gptj.alpaca.gpt4': 'vicgalle/gpt-j-6B-alpaca-gpt4',
        'gptj.alpaca': 'bertin-project/bertin-gpt-j-6B-alpaca',
        'oa.galactia.6.7b': 'OpenAssistant/galactica-6.7b-finetuned',
        'opt6.7b': 'facebook/opt-6.7b',
        'llama': 'decapoda-research/llama-7b-hf',
        'vicuna.13b': 'lmsys/vicuna-13b-delta-v0',
        'vicuna.7b': 'lmsys/vicuna-7b-delta-v0',
        'llama-trl': 'trl-lib/llama-7b-se-rl-peft',
        'opt.nerybus': 'KoboldAI/OPT-6.7B-Nerybus-Mix',

        # # > 7B models
        'oa.pythia.12b': 'OpenAssistant/oasst-sft-1-pythia-12b',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b',
        'gpt13b': 'cerebras/Cerebras-GPT-13B'
        
            }


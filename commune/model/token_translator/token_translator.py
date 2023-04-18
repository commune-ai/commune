import commune
import torch
from typing import Dict, Any, List, Tuple
import streamlit as st
commune.new_event_loop()
from commune.utils.tokenizer import get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases, \
        encode_topk, decode_topk, prep_tokenizer

class TokenTranslator(commune.Module):
    def __init__(self, from_tokenizer='facebook/opt-6.7b', to_tokenizer='gpt2'):
        
        self.set_translator(from_tokenizer=from_tokenizer, 
                                to_tokenizer=to_tokenizer)
        


    def set_translator(self, from_tokenizer, to_tokenizer):
        
        self.from_tokenizer  = self.get_tokenizer(from_tokenizer)
        self.to_tokenizer  = self.get_tokenizer(to_tokenizer)
        
        self.from_tokenizer = prep_tokenizer(self.from_tokenizer, self.to_tokenizer)
        
        self.to_translation_map = self.get_translation_map(self.from_tokenizer, self.to_tokenizer)
        self.from_translation_map = self.get_translation_map(self.to_tokenizer, self.from_tokenizer)
        self.split_map_cache = {}
        
        
    # def translate_logits(self, logits: torch.FloatTensor,
    #                               tokens: torch.LongTensor, tokens_std: torch.LongTensor):
    #     probs_std = translate_logits_to_probs_std(logits=logits,
    #                                                 offset_mapping=tokens['offset_mapping'], offset_mapping_std=tokens['offset_mapping_std'],
    #                                                 tokenizer=self.from_tokenizer, std_tokenizer=self.to_tokenizer,
    #                                                 split_map_cache=self.split_map_cache,
    #                                                 to_translation_map=self.to_translation_map, from_translation_map=self.from_translation_map,
    #                                                 tokens=tokens['input_ids'], tokens_std=tokens_std)
    #     probs_std = probs_std
    #     logits_std = torch.log(probs_std + 1e-40)
        
    #     return logits_std
    
    def translate_tokens(self, tokens: torch.LongTensor):
        return self.map_tokens(tokens, to_tokenizer=self.to_tokenizer).input_ids
        


    def map_tokens(self, token_batch, to_tokenizer=None, return_offsets_mapping=True):
        r""" Tokenizer remapping; decodes the message and then remaps the message using a new tokenizer
            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    token_batch to be retokenized, [batch_size, sequence_len]
                to_tokenizer ( :obj:`transformers.Tokenizer`, `optional`):
                    The standard tokenizer which was used to tokenize the input.
                return_offsets_mapping ( :obj:`bool`, `required`):
                    Return offsets_mapping in tokenization to delineate token segment positions.
        """
        if to_tokenizer is None:
            to_tokenizer = self.to_tokenizer

        text_batch = to_tokenizer.batch_decode(token_batch)  # decode tokens to original text
        result = translate_special_token_text(text_batch, to_tokenizer, self.from_tokenizer)  # translate special tokens
        to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch = result

        tokens = self.to_tokenizer(to_text_batch, padding=True, truncation=True, max_length=token_batch.size(1), return_tensors='pt',
                                add_special_tokens=False)  # assume tokenizer.padding_side = 'left'

        if return_offsets_mapping:  # get offsets_mapping in tokenization to delineate token segment positions
            server_tokens = self.from_tokenizer(to_text_batch, return_offsets_mapping=True, add_special_tokens=False)
            std_tokens = to_tokenizer(text_batch, return_offsets_mapping=True)  # encode again to get offsets mapping

            # pad offsets so that special token offset widths match for continued correct alignment
            tokens['offset_mapping'] = pad_offsets(server_tokens['offset_mapping'], to_offsets_batch, pad_offsets_batch)
            tokens['offset_mapping_std'] = pad_offsets(std_tokens['offset_mapping'], from_offsets_batch,
                                                       pad_offsets_batch)
        return tokens

    

    
    tokenizer_cache = {}
    def get_tokenizer(cls, tokenizer_name: str, cache:bool = True) -> 'PreTrainedTokenizerBase':
        from transformers import AutoTokenizer
        r"""
        Returns a tokenizer instance for a given tokenizer name.
            Args:
                tokenizer_name (:obj:`str`, `required`):
                    Name of the tokenizer to be loaded.
            Returns:
                tokenizer (:obj:`PreTrainedTokenizerBase`):
                    A tokenizer instance.
        """
        tokenizer = None
        if cache:
            if tokenizer_name in cls.tokenizer_cache:
                tokenizer = cls.tokenizer_cache[tokenizer_name]
            else:
                tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)

        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        
        
        return tokenizer
    


    @classmethod
    def get_translation_map(cls, from_tokenizer: 'PreTrainedTokenizerBase',
                            to_tokenizer: 'PreTrainedTokenizerBase') -> Dict[str, Any]:
        r"""
        Map individual token phrases from a tokenizer to another tokenizer.
            Args:
                from_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    From tokenizer.
                to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    To tokenizer.

            Returns:
                translation_map (:obj:`Dict[str, Any]`, `required`):
                    Maps for each observed length, a source token to a token sequence of that length,
                    with source index to target indices.
        """

        translation_map = {}

        phrases = from_tokenizer.batch_decode(range(from_tokenizer.vocab_len))  # tokens to strings
        # st.write(phrases[:100])
        to_tokens = to_tokenizer(phrases)['input_ids']  # convert single token from-phrases to to-tokenization
        

            
        translation_map = {}
        counts = {}
        for from_idx, to_idx in enumerate(to_tokens):
            to_idx_len = len(to_idx)
            
            if to_idx_len not in translation_map:
                translation_map[to_idx_len] = {
                    'from': [],
                    'to': []
                }
            
            translation_map[to_idx_len]['from'].append(from_idx)
            translation_map[to_idx_len]['to'].append(to_idx)
            
        for to_idx_len in translation_map.keys():
            for k in ['from', 'to']:
                translation_map[to_idx_len][k] = torch.LongTensor(translation_map[to_idx_len][k])
        
        return translation_map
                
        
    def map_logits(self, logits: torch.FloatTensor) -> None:
        r"""
        Translate a single token probability distribution from a source tokenization to a
        sequence of probability distributions over a target tokenization.
            Args:
                probs_from (:obj:`torch.FloatTensor`, `required`):
                    [vocab_size] Input probability distribution over a from-tokenizer vocabulary.
                probs_to (:obj:`torch.FloatTensor`, `required`):
                    [many, vocab_size] Output probability distributions over a to-tokenizer vocabulary.
                translation_map (:obj:`Dict[str, Any]`, `required`):
                    Maps for each observed length, a source token to a token sequence of that length,
                    with source index to target indices.

            Returns:

        """
        
        assert logits.dim() == 3, f'Expected logits to be 3D, got {logits.dim()}D'
        batch_size, seqeunce_length, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        
        to_vocab_size = self.to_tokenizer.vocab_len
        
        translation_map =   self.to_translation_map
        to_logits = torch.zeros(logits.shape[0],to_vocab_size).to(logits.device)  # [vocab_size] 

        # === Unroll single distribution into std sequence ===
        
        probs = torch.softmax(logits, dim=-1)  # [vocab_size, subset_size_std]
        to_probs = torch.full_like(to_logits, 1e-8)  # [vocab_size, subset_size_std]
        counts = torch.full_like(to_logits, 1e-8)
        for map_len in translation_map.keys():  # each one-to-many mapping length available
            
            # map_len = int(map_len)
        
            to_idx = translation_map[map_len]['to'].T  # [map_len, subset_size_std]
            
            from_idx = translation_map[map_len]['from']


            for i in range(len(to_idx)):
                # to_probs[:, to_idx[i]] += probs[:, from_idx]
                # counts[:, to_idx[i]] += torch.ones_like(counts[:, to_idx[i]])
                to_probs[:, to_idx[i]] += probs[:, from_idx]
                counts[:, to_idx[i]] += torch.ones_like(probs[:, from_idx])
                # add probs in-place
        # to_probs = to_probs / counts
        to_probs = to_probs / to_probs.sum(dim=-1, keepdim=True)
        # self.print(to_probs.sum(dim=-1, keepdim=True))
        to_logits = torch.log(to_probs)
        
        to_logits =  to_logits.reshape(batch_size, seqeunce_length, to_vocab_size)
        
        
        return to_logits
    @classmethod
    def set_vocab_len(cls, tokenizer: 'PreTrainedTokenizerBase'):
        r"""
        Sets the tokenizer.vocab_len if unset, to store the real vocabulary size according to the vocab or encoder.
            Args:
                tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    Tokenizer to set vocab_len for.
            Returns:

        """
        if not hasattr(tokenizer, 'vocab_len'):
            if hasattr(tokenizer, 'vocab'):  # use independent vocab_len when tokenizer.vocab_size != len(tokenizer.vocab)
                tokenizer.vocab_len = len(tokenizer.vocab)
            elif hasattr(tokenizer, 'encoder'):  # tokenizers like facebook/opt-* has encoder=vocab
                tokenizer.vocab_len = len(tokenizer.encoder)
            else:  # revert to vocab_size
                tokenizer.vocab_len = tokenizer.vocab_size

    @classmethod
    def prep_tokenizer(cls, tokenizer, to_tokenizer=None):
        tokenizer.padding_side = "left"  # Generative default expects most recent token on right-hand side with padding on left. https://github.com/huggingface/transformers/pull/10552
        # tokenizer.add_prefix_space = False
        # tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
        # tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
        # tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
        # tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
        # tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
        # tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
        # tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
        # additional_special_tokens = [
        #     "<s>NOTUSED",  # Used by BARThez
        #     "</s>NOTUSED", # Used by BARThez
        #     "<eop>", # Used by MarianMT
        #     "<eod>", # Used by MarianMT
        #     "<formula>", # Used by Transformer XL
        #     "<mask_1>" # Used by Pegasus
        #     "<special0>", # Used by XLM
        #     "<special1>", # Used by XLM
        #     "<special2>", # Used by XLM
        #     "<special3>", # Used by XLM
        #     "<special4>", # Used by XLM
        #     "<special5>", # Used by XLM
        #     "<special6>", # Used by XLM
        #     "<special7>", # Used by XLM
        #     "<special8>", # Used by XLM
        #     "<special9>", # Used by XLM
        # ]
        # tokenizer.additional_special_tokens = additional_special_tokens

        # Define PAD Token = EOS Token (GPT2 generate convention, when PAD Token is None)
        # https://github.com/huggingface/transformers/blob/49c8c67fb815a277405f84dea4a66353e19fb347/tests/models/gpt2/test_modeling_gpt2.py#L532
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        cls.set_vocab_len(tokenizer)
        cls.set_whitespace_preserving(tokenizer)

        if to_tokenizer is not None:
            set_std_token_phrases(tokenizer, to_tokenizer)

        return tokenizer
    

    def tokenize(self, 
                 text: str = 'Whadup',
                 padding=True, 
                 truncation=True, 
                 max_length=256,
                 return_tensors='pt',
                 add_special_tokens=False,
                 device:str = None, 
                tokenizer = 'from',
                 **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        if tokenizer is None:
            tokenizer = 'from'
        tokenizer = getattr(self, f'{tokenizer}_tokenizer')
        sample = tokenizer(text, 
                                             padding=padding, 
                                             truncation=truncation, 
                                             max_length=max_length, 
                                             return_tensors=return_tensors,
                                             add_special_tokens=add_special_tokens, 
                                             **kwargs)  # assume tokenizer.padding_side = 'left'

        
        sample = dict(
            input_ids= sample['input_ids'],
            attention_mask= sample['attention_mask']
        )
        
        return sample



    def detokenize(self,input_ids: torch.Tensor,
                   tokenizer= 'from', 
                   **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        if tokenizer is None:
            tokenizer = 'from'
        tokenizer = getattr(self, f'{tokenizer}_tokenizer')
        text = tokenizer.batch_decode(input_ids,**kwargs)  # assume tokenizer.padding_side = 'left'

        return text



    @staticmethod
    def set_whitespace_preserving(tokenizer: 'PreTrainedTokenizerBase'):
        r"""
        Sets the tokenizer.whitespace_preserving if unset, indicates if tokenizer preserves whitespace like GPT-style,
        or not like BERT-style.
            Args:
                tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    Tokenizer to set vocab_len for.
            Returns:

        """
        if not hasattr(tokenizer, 'whitespace_preserving'):
            space_token = tokenizer(' ', add_special_tokens=False)['input_ids']
            space_text = tokenizer.decode(space_token)
            if space_text == ' ':
                tokenizer.whitespace_preserving = True
            else:
                tokenizer.whitespace_preserving = False




    @classmethod
    def test(cls, model='model.gpt2.7b', dataset='dataset.text.bittensor'):
        cls.print('test')
        dataset = commune.connect(dataset)

        model = commune.connect(model, wait_for_server=True)
        model.set_model()
        self = cls(from_tokenizer = model.tokenizer_name(), to_tokenizer = 'gpt2')
        
        loss_fn = commune.get_module('model.transformer').calculate_loss

        for i in range(100):
            sample = dataset.sample(batch_size=8, sequence_length=64, no_tokenizer=False)
            og_sample = cls.copy(sample)
            
            
            sample['map_tokens'] = False
            sample['map_logits'] = False
            sample['train'] = True
            og_tokens = sample['input_ids']
            sample['input_ids'] =  self.map_tokens(sample['input_ids']).input_ids
            output = model.forward(**sample)
            
            
            # cls.print(output['stats'])
            
            output['logits'] = decode_topk(output['topk'], vocab_size=self.from_tokenizer.vocab_len)
            # output['logits'] = self.translate_logits( output['logits'], tokens, sample['input_ids'])
            # sample['input_ids']= og_tokens

            output.update(sample)
            cls.print(loss_fn(**output))
            
        
    def translate_logits(self, logits: torch.FloatTensor) -> None:
        r"""
        Translate a single token probability distribution from a source tokenization to a
        sequence of probability distributions over a target tokenization.
            Args:
                probs_from (:obj:`torch.FloatTensor`, `required`):
                    [vocab_size] Input probability distribution over a from-tokenizer vocabulary.
                probs_to (:obj:`torch.FloatTensor`, `required`):
                    [many, vocab_size] Output probability distributions over a to-tokenizer vocabulary.
                translation_map (:obj:`Dict[str, Any]`, `required`):
                    Maps for each observed length, a source token to a token sequence of that length,
                    with source index to target indices.

            Returns:

        """
        
        assert logits.dim() == 3, f'Expected logits to be 3D, got {logits.dim()}D'
        batch_size, seqeunce_length, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        
        to_vocab_size = self.to_tokenizer.vocab_len
        
        translation_map =   self.to_translation_map
        to_logits = torch.zeros(logits.shape[0],to_vocab_size).to(logits.device)  # [vocab_size] 

        # === Unroll single distribution into std sequence ===
        
        probs = torch.softmax(logits, dim=-1)  # [vocab_size, subset_size_std]
        to_probs = torch.full_like(to_logits, 1e-8)  # [vocab_size, subset_size_std]
        counts = torch.full_like(to_logits, 1e-8)
        for map_len in translation_map.keys():  # each one-to-many mapping length available
            
            # map_len = int(map_len)
        
            to_idx = translation_map[map_len]['to'].T  # [map_len, subset_size_std]
            
            from_idx = translation_map[map_len]['from']


            for i in range(len(to_idx)):
                # to_probs[:, to_idx[i]] += probs[:, from_idx]
                # counts[:, to_idx[i]] += torch.ones_like(counts[:, to_idx[i]])
                to_probs[:, to_idx[i]] += probs[:, from_idx]
                counts[:, to_idx[i]] += torch.ones_like(probs[:, from_idx])
                # add probs in-place
        # to_probs = to_probs / counts
        to_probs = to_probs / to_probs.sum(dim=-1, keepdim=True)
        # self.print(to_probs.sum(dim=-1, keepdim=True))
        to_logits = torch.log(to_probs)
        
        to_logits =  to_logits.reshape(batch_size, seqeunce_length, to_vocab_size)
        
        
        return to_logits
    
    
if __name__ == "__main__":
    TokenTranslator.run()
    
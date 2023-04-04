

import commune
from typing import *
from transformers import PreTrainedTokenizerBase
import torch
from commune.utils.tokenizer import prep_tokenizer, get_translation_map, translate_special_token_text
class TokenizerMap(commune.Module):
    shortcuts =  {
        'gptj': 'EleutherAI/gpt-j-6b',
        'gpt2.7b': 'EleutherAI/gpt-neo-2.7B',
        'gpt125m': 'EleutherAI/gpt-neo-125M',
        'gptjt': 'togethercomputer/GPT-JT-6B-v1',
        'gptneox': 'EleutherAI/gpt-neox-20b',
        'gpt20b': 'EleutherAI/gpt-neox-20b',
        'opt13b': 'facebook/opt-13b'

         }
    def __init__(self, tokenizer='gpt20b', std_tokenizer=None, *args, **kwargs):
        
        self.set_tokenizer(tokenizer=tokenizer, std_tokenizer=std_tokenizer)
        
        

    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer', None], std_tokenizer=None):
        
        from transformers import AutoTokenizer
        if isinstance(tokenizer, str):
            tokenizer = self.shortcuts.get(tokenizer, tokenizer)
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast= True)
            except ValueError:
                print('resorting ot use_fast = False')
                tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.tokenizer = tokenizer
        
        
        self.std_tokenizer = std_tokenizer if std_tokenizer else self.default_std_tokenizer()
        self.tokenizer = self.prep_tokenizer(self.tokenizer, self.std_tokenizer)
        
        self.to_translation_map = self.get_translation_map(self.tokenizer, self.std_tokenizer)
        self.from_translation_map = self.get_translation_map(self.std_tokenizer, self.tokenizer)
        self.split_map_cache = {}

        return self.tokenizer
    @classmethod
    def default_std_tokenizer(cls):
        try:
            import bittensor
        except RuntimeError:
            commune.new_event_loop()
            import bittensor
        return bittensor.tokenizer()
    
    @classmethod
    def prep_tokenizer(cls, tokenizer, std_tokenizer=None):
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
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        cls.set_vocab_len(tokenizer)
        cls.set_whitespace_preserving(tokenizer)

        if std_tokenizer is not None:
            cls.set_std_token_phrases(tokenizer, std_tokenizer)

        return tokenizer


    @staticmethod
    def set_vocab_len(tokenizer: PreTrainedTokenizerBase):
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


    @staticmethod
    def set_whitespace_preserving(tokenizer: PreTrainedTokenizerBase):
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
    def set_std_token_phrases(cls, tokenizer, std_tokenizer):
        r"""
        Sets std_token_phrases which are the tokenizer token strings tokenized with std_tokenizer, so
        the std_tokenizer equivalent of the tokenizer token strings.
        Used for converting model predictions/logits into std_tokenizer representations, for example in TextCausalLMNext.
            Args:
                tokenizer(:obj:`PreTrainedTokenizerBase`, `required`):
                    Tokenizer to set std_token_phrases for.
                std_tokenizer(:obj:`PreTrainedTokenizerBase`, `required`):
                    Standard bittensor tokenizer to convert to.

            Returns:

        """
        # === Tokenizer phrases to memory ===
        if not hasattr(tokenizer, 'phrases'):
            if tokenizer.whitespace_preserving:
                tokenizer.phrases = tokenizer.batch_decode(range(tokenizer.vocab_len))  # server tokens to strings
            else:
                tokenizer.phrases = [' ' + phrase for phrase in
                                    tokenizer.batch_decode(range(tokenizer.vocab_len))]  # server tokens to strings

        if not hasattr(tokenizer, 'std_token_phrases'):
            # Retokenize phrases to new tokenizer
            tokenizer.std_token_phrases = std_tokenizer(tokenizer.phrases)['input_ids']  # [topk, max_len] convert phrases to tokens sequences


    @classmethod
    def get_translation_map(cls, from_tokenizer: PreTrainedTokenizerBase,
                            to_tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
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
        cls.set_vocab_len(from_tokenizer)
        cls.set_vocab_len(to_tokenizer)

        translation_map = {'lengths': {}}

        phrases = from_tokenizer.batch_decode(range(from_tokenizer.vocab_len))  # tokens to strings

        to_tokens = to_tokenizer(phrases)['input_ids']  # convert single token from-phrases to to-tokenization
        to_tokens_lens = [len(p) for p in to_tokens]
        unique_lens = set(to_tokens_lens)
        max_len = max(unique_lens)
        counts = torch.zeros((max_len, to_tokenizer.vocab_len), dtype=torch.long)

        for l in unique_lens:  # each unique one-to-many mapping length
            from_idx = [i for i, k in enumerate(to_tokens_lens) if k == l]  # find len l to-tokenizations
            subset = [to_tokens[i] for i in from_idx]  # find len l to-tokenizations
            from_idx = torch.tensor(from_idx, dtype=torch.long)  # [subset_size]
            to_idx = torch.tensor(subset, dtype=torch.long)  # [subset_size, l]
            translation_map['lengths'][l] = {'from': from_idx,
                                            'to': to_idx}
            # accumulate counts on tokens, to be used to divide probability mass over its channeled sequences
            counts[:l, :].scatter_add_(1, to_idx.T, torch.ones((l, len(subset)), dtype=torch.long))

        translation_map['counts'] = counts
        return translation_map


    def translate_one_to_many(probs_from: torch.FloatTensor, probs_to: torch.FloatTensor,
                            translation_map: Dict[str, Any]) -> None:
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
        many_len = probs_to.shape[0]

        # === Unroll single distribution into std sequence ===
        for i in range(many_len):  # each unrolling step
            for map_len in translation_map['lengths'].keys():  # each one-to-many mapping length available
                if map_len < i + 1:
                    continue  # skip unrolling steps not available in a shorter mapping length
                from_idx = translation_map['lengths'][map_len]['from']
                to_idx = translation_map['lengths'][map_len]['to'].T  # [map_len, subset_size_std]
                probs_to[i, :].scatter_add_(0, to_idx[i, :], probs_from[from_idx])  # add probs in-place


        @staticmethod
        def translate_many_to_one(probs_from: torch.FloatTensor, probs_to: torch.FloatTensor,
                                translation_map: Dict[str, Any]) -> None:
            r"""
                Translate a sequence of token probability distributions from a source tokenization to a
                single token probability distribution over a target tokenization.
                    Args:
                        probs_from (:obj:`torch.FloatTensor`, `required`):
                            [many, vocab_size] Input probability distributions over a from-tokenizer vocabulary.
                        probs_to (:obj:`torch.FloatTensor`, `required`):
                            [vocab_size] Output probability distribution over a to-tokenizer vocabulary.
                        translation_map (:obj:`Dict[str, Any]`, `required`):
                            Maps for each observed length, a source token to a token sequence of that length,
                            with source index to target indices.

                    Returns:

                """
            many_len = probs_from.shape[0]
            probs_from_copy = probs_from.clone()  # will modify from-probabilities

            # === Spread probability mass over realized sequences ===
            counts = translation_map['counts']  # [max_len, vocab_size]
            translation_max_len = counts.shape[0]  # maximum possible many-to-one length available in translation map

            if many_len <= translation_max_len:
                probs_from_copy /= counts[:many_len, :]  # divide probability mass by amount of paths crossing each token
            else:  # limit probs_from token depth to max_len
                probs_from_copy[:translation_max_len, :] /= counts

            # === Reverse map std token to source sequences, gather avg. sequence prob ===
            for map_len in translation_map['lengths'].keys():  # mutually exclusive over std tokens
                from_idx = translation_map['lengths'][map_len]['from']  # [subset_size_std] one std token
                to_idx = translation_map['lengths'][map_len]['to'].T  # [map_len, subset_size_std] many server token seq
                if many_len < map_len:  # sequence beyond segment_count has min probability 0
                    to_idx = to_idx[:many_len, :]  # [segment_count, subset_size_std]
                server_seq_tokens = probs_from_copy.gather(1, to_idx)  # [map_len, subset_size_std] gather sequences
                probs_to[from_idx] = server_seq_tokens.sum(dim=0) / map_len  # [subset_size_std] in-place average approx.

        @staticmethod
        def translate_tokenizer_probs(probs: torch.FloatTensor, probs_std: torch.FloatTensor,
                                    offset_mapping: List[tuple], offset_mapping_std: List[tuple],
                                    tokenizer: PreTrainedTokenizerBase, std_tokenizer: PreTrainedTokenizerBase,
                                    split_map_cache: Dict[tuple, List[Dict[str, torch.Tensor]]],
                                    to_translation_map: Dict[str, Any], from_translation_map: Dict[str, Any],
                                    tokens: torch.LongTensor, tokens_std: torch.LongTensor) -> None:
            r"""
            Translates source token probability distributions to target probability distributions, by
            aligning segments through source token splits, then greedily performing one-to-one,
            one-to-many, many-to-one distribution mappings.
                Args:
                    probs (:obj:`torch.FloatTensor`, `required`):
                        [sequence_len, vocab_size] Input probability distribution over a source tokenizer vocabulary.
                    probs_std (:obj:`torch.FloatTensor`, `required`):
                        [std_sequence_len, std_vocab_size] Output probability distribution over a target tokenizer vocabulary.
                        Reference that will be written in-place.
                    offset_mapping (:obj:`List[tuple]`, `required`):
                        Tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...].
                    offset_mapping_std (:obj:`List[tuple]`, `required`):
                        Standard tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...]
                    tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                        Source tokenizer.
                    std_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                        Standard/target tokenizer.
                    split_map_cache (:obj:`Dict[tuple, List[Dict[str, torch.Tensor]]]`, `required`):
                        A dictionary of depths keying split_maps of mappings from original tokens to
                        target tokens at each depth of the split. Adds split_maps to cache for faster future recall.
                    tokens (:obj:`torch.LongTensor`, `required`):
                        [sequence_len] A sequence of tokens produced by the source tokenizer.
                    tokens_std (:obj:`torch.LongTensor`, `required`):
                        [std_sequence_len] A sequence of tokens produced by the standard tokenizer.
                    to_translation_map (:obj:`Dict[str, Any]`, `required`):
                        Maps for each observed length, a source token to a token sequence of that length,
                        with source index to target indices.
                    from_translation_map (:obj:`Dict[str, Any]`, `required`):
                        Maps for each observed length, a source token to a token sequence of that length,
                        from target index to source indices.

                Returns:

            """
            # === Align tokenized sequences via source token splitting ===
            result = align_tokenizer_sequences(probs, offset_mapping, offset_mapping_std,
                                            tokenizer, split_map_cache, tokens.cpu(), tokens_std.cpu())
            aligned_probs, aligned_offset_mapping, aligned_tokens = result

            # === Get one-to-many / many-to-one mappings ===
            mappings = get_tokenizer_sequence_mappings(aligned_offset_mapping, offset_mapping_std)

            # === Perform probability mappings ===
            for (right_idx, right_idx_std, segment_count_base, segment_count_std_base,
                segment_count_overlap, segment_count_std_overlap) in mappings[1:]:  # don't map start token

                segment_count = segment_count_base + segment_count_overlap  # calculate effective segments length
                segment_count_std = segment_count_std_base + segment_count_std_overlap  # calculate effective segments length

                # === One-to-many / one-to-one mapping ===
                if segment_count_base == 1:
                    start_idx_std = right_idx_std - segment_count_std  # calculate starting index

                    translate_one_to_many(aligned_probs[right_idx-1],
                                        probs_std[start_idx_std:start_idx_std+segment_count_std],
                                        to_translation_map)

                # === Many-to-one mapping ===
                elif segment_count_std_base == 1:  # many-to-one
                    start_idx = right_idx - segment_count  # calculate starting index

                    translate_many_to_one(aligned_probs[start_idx:right_idx],
                                        probs_std[right_idx_std-1],
                                        from_translation_map)

                else:
                    print('Undefined mapping.')

    @classmethod
    def encode_topk(cls,  forward_response_tensor: torch.Tensor , topk:int=4096) -> torch.Tensor:
    
        """ Returns topk tokens/probabilities given unnormalized logits as input. """

        #import ipdb; ipdb.set_trace()

        logits = forward_response_tensor  # unnormalized logit scores: [batch_size, sequence_len, vocab_size]
        probs = torch.softmax(logits, dim=-1).to(torch.float32)  # normalized probabilities: [batch_size, sequence_len, vocab_size]

        topk_indices = torch.argsort(probs, dim=-1, descending=True)[...,:topk]
        # topk_values, topk_indices = torch.topk(probs, topk) # topk probs and indices: [batch_size, sequence_len, topk]

        topk_values = probs.gather( index=topk_indices, dim=-1)
        encoded_probs = torch.cat([topk_values, topk_indices], dim=-1)  # [batch_size, sequence_len, topk + topk]
        return encoded_probs  # [batch_size, sequence_len, topk + topk]


    @classmethod
    def decode_topk(cls,  forward_response_tensor: torch.Tensor, topk:int=4096, vocab_size:int=50400) -> torch.Tensor:
        """ Returns full logits by decoding topk-encoding input. """
        batch_size, sequence_len, _ = forward_response_tensor.shape
        encoded_probs = forward_response_tensor  # encoded probabilities: [batch_size, sequence_len, topk + topk]
        topk_values = encoded_probs[..., :topk]  # topk probs: [batch_size, sequence_len, topk]
        topk_indices = encoded_probs[..., topk:].long()  # topk probs indices: [batch_size, sequence_len, topk]

        topk_pmass = topk_values.sum(dim=-1)  # topk probability mass: [batch_size, sequence_len]
        remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # remainder probability mass: [batch_size, sequence_len]
        
        remainder_floor = remainder_pmass / (vocab_size - topk)  # divide remainder: [batch_size, sequence_len]

        logits = torch.ones((batch_size, sequence_len, vocab_size), dtype=topk_values.dtype).to(topk_values.device)
        logits *= torch.log(remainder_floor)[:, :, None]  # set probability floor: [batch_size, sequence_len, vocab_size]

        logits.scatter_(-1, topk_indices, torch.log(topk_values + 1e-40))  # insert topk probs: [batch_size, sequence_len, vocab_size]

        return logits  # [batch_size, sequence_len, vocab_size]

    @classmethod
    def calculate_loss( cls, pred, gt):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt)
        return loss

    def tokenize(self, text,**kwargs):
        default_kwargs = dict( return_tensors='pt', 
                              max_length=256, 
                              truncation=True, 
                              padding='max_length')
        kwargs = {**default_kwargs, **kwargs}
        return self.tokenizer(text, **kwargs)
    

    def token_remap(self, token_batch, return_offsets_mapping=False):
        r""" Tokenizer remapping; decodes the message and then remaps the message using a new tokenizer
            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    token_batch to be retokenized, [batch_size, sequence_len]
                std_tokenizer ( :obj:`transformers.Tokenizer`, `optional`):
                    The standard tokenizer which was used to tokenize the input.
                return_offsets_mapping ( :obj:`bool`, `required`):
                    Return offsets_mapping in tokenization to delineate token segment positions.
        """

        text_batch = self.tokenizer.batch_decode(token_batch)  # decode tokens to original text
        result = translate_special_token_text(text_batch, self.std_tokenizer, self.tokenizer)  # translate special tokens
        to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch = result

        tokens = self.std_tokenizer(to_text_batch, padding=True, truncation=True, max_length=token_batch.size(1), return_tensors='pt',
                                add_special_tokens=False)  # assume tokenizer.padding_side = 'left'

        if return_offsets_mapping:  # get offsets_mapping in tokenization to delineate token segment positions
            server_tokens = self.tokenizer(to_text_batch, return_offsets_mapping=True, add_special_tokens=False)
            std_tokens = self.std_tokenizer(text_batch, return_offsets_mapping=True)  # encode again to get offsets mapping

            # pad offsets so that special token offset widths match for continued correct alignment
            tokens['offset_mapping'] = pad_offsets(server_tokens['offset_mapping'], to_offsets_batch, pad_offsets_batch)
            tokens['offset_mapping_std'] = pad_offsets(std_tokens['offset_mapping'], from_offsets_batch,
                                                       pad_offsets_batch)
        return tokens
 
    @classmethod
    def get_special_token_pairings(cls, from_tokenizer: PreTrainedTokenizerBase,
                                to_tokenizer: PreTrainedTokenizerBase) -> Dict[str, str]:
        r"""
        Determines a prioritized matching of special token texts between two tokenizers.
        Purpose is to produce replacement pairs so special token test is correctly represented for target tokenizer.
            Args:
                from_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    From tokenizer.
                to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    To tokenizer.

            Returns:
                pairings (:obj:`Dict[str, str]`, `required`):
                    Prioritized dictionary of From_special_token_text -> To_special_token_text.
        """
        pairings = {}

        # some tokenizers e.g. GPT2 have the same text signifying BOS and EOS, while in other e.g. XGLM they differ
        # so prioritize EOS token first, since this seems to be the default context separator, e.g. XGLM, GerPT2, GPT2
        if ('eos_token' in from_tokenizer.special_tokens_map) and ('eos_token' in to_tokenizer.special_tokens_map):
            pairings[getattr(from_tokenizer, 'eos_token')] = getattr(to_tokenizer, 'eos_token')

        for special_token in from_tokenizer.special_tokens_map:
            if special_token in to_tokenizer.special_tokens_map:
                if getattr(from_tokenizer, special_token) not in pairings:  # prevent priority overwrite
                    pairings[getattr(from_tokenizer, special_token)] = getattr(to_tokenizer, special_token)

        return pairings
   
    @classmethod
    def translate_special_token_text(cls, text_batch: List[str], from_tokenizer: PreTrainedTokenizerBase,
                                    to_tokenizer: PreTrainedTokenizerBase) -> Tuple[List[str],
                                                                                    List[List[List[int]]],
                                                                                    List[List[List[int]]],
                                                                                    List[List[List[Any]]]]:
        r"""
        Translates special_token signifier text in from_tokenizer to to_tokenizer special_token text, for
        a given text_batch. Resulting to_text_batch can then be to_tokenized where special_tokens should
        map to its single corresponding token, despite signifier text difference compared to from_tokenizer.
            Args:
                text_batch (:obj:`List[str]`, `required`):
                    List of strings to translate special tokens for.
                from_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    From tokenizer.
                to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    To tokenizer.

            Returns:
                to_text_batch (:obj:`List[str]`, `required`):
                    List of strings where special text has been replaced.
                from_offsets_batch (:obj:`List[List[List[int]]]`, `required`):
                    Batch of tokenizer offset mappings selecting replacement tuples in from_tokenizer text
                        [[(left_0, right_0), (left_1, right_1), ...], ...].
                to_offsets_batch (:obj:`List[List[List[int]]]`, `required`):
                    Batch of tokenizer offset mappings selecting replacement tuples in to_tokenizer text
                        [[(left_0, right_0), (left_1, right_1), ...], ...].
                pad_offsets_batch (:obj:`List[List[List[Any]]]`, `required`):
                    Batch of offset paddings associated with each replacement tuple
                        [[(left_pad_0, right_pad_0), (left_pad_1, right_pad_1), ...], ...].
        """
        to_text_batch = []
        from_offsets_batch = []
        to_offsets_batch = []
        pad_offsets_batch = []

        # === Get special-token text replacement pairs ===
        pairings = cls.get_special_token_pairings(from_tokenizer, to_tokenizer)

        for text in text_batch:
            from_offsets = []
            padding_offsets = []
            for token_string in pairings:
                offsets = find_offsets(text, token_string)  # find special-token locations
                from_offsets += [[left, right, pairings[token_string]] for left, right in offsets]

                pad_string = token_string if len(token_string) > len(pairings[token_string]) else pairings[token_string]
                padding_offsets += [[left, right, pad_string] for left, right in offsets]

            from_offsets = sorted(from_offsets)  # incrementally arrange locations
            to_text, to_offsets = replace_at_offsets(text, from_offsets)  # replace special-token text
            pad_text, padding_offsets = replace_at_offsets(text, padding_offsets)  # pad special-token text locations

            to_text_batch += [to_text]
            from_offsets_batch += [[[left, right] for left, right, _ in from_offsets]]
            to_offsets_batch += [to_offsets]
            pad_offsets_batch += [padding_offsets]

        return to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch

        
    @classmethod
    def test(cls, topk=4096, output_length=20):
        
        model = commune.connect('model::gpt125m')
        dataset = commune.connect('dataset::bittensor')
        
        
        sample = dataset.sample(no_tokenizer=True)
        tokenizer = cls(tokenizer = 'gpt20b')
        sample = tokenizer.tokenize(sample['text'])
        
        
        

        sample.update(dict(
            output_hidden_states=True,
            hidden_dim_bounds = [0, 100],
            output_logits=False, 
            output_topk=True, 
            output_length=output_length,
            token_remap = False , 
            logit_remap = False,
            topk=topk
        ))
        sample['input_ids'] = tokenizer.token_remap(sample['input_ids'])['input_ids']
        import streamlit as st
        st.write(sample['input_ids'])
        targets = sample['input_ids'][:,1:]
        sample['input_ids'] = sample['input_ids'][:,:-1]
        pred = model.forward(**sample, no_grad=True)
        
        pred['logits'] = cls.decode_topk(pred['topk'])
        
        logits =  pred['logits']
        import streamlit as st
        gt = targets[:,-logits.shape[1]:].flatten()
        pred = logits.reshape(-1, logits.size(-1))
        loss = cls.calculate_loss(pred=pred, 
                                    gt=gt)              
        
        
        st.write(loss)
        # output['logits'] = decode_topk(output['topk'])
        
        # print(cls.calculate_loss(output['logits'].reshape(-1, output['logits'].shape[-1]), targets[:, -output_length:].flatten()))
     

    @classmethod
    def calculate_loss( cls, pred, gt):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss =  loss_fn(pred, gt)
        return loss


if __name__ == '__main__':
    TokenizerMap.test()
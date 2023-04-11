import commune
class TokenMapper(commune.Module):
    def __init__(self, from_tokenizer='gpt2', to_tokenizer='facebook/opt-6.7b'):
        self.set_tokenizer_pair(tokenizer1, tokenizer2)

    def set_tokenizer_pair(self, from_tokenizer, to_tokenizer):
        
        self.from_tokenizer  = self.get_tokenizer(from_tokenizer)
        self.to_tokenizer  = self.get_tokenizer(to_tokenizer)
        self.translation_map = self.get_translation_map(self.from_tokenizer, self.to_tokenizer)
        
        
    def forward(self, logits, from_tokenizer, to_tokenizer):
        logits = self.map_logits_between_tokenizers(logits, self.tokenizers[from_tokenizer], self.tokenizers[to_tokenizer])
        return logits
    
    
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

        if cache:
            if tokenizer_name in cls.tokenizer_cache:
                tokenizer =  AutoTokenizer.from_pretrained(tokenizer_name)
                cls.tokenizer_cache[tokenizer_name] = tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        return tokenizer
    
    @classmethod
    def map_logits_between_tokenizers(cls, logits, tokenizer1, tokenizer2):
        # Convert logits to probabilities using softmax function
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probabilities = softmax(logits)

        # Map the token ids and probabilities between the tokenizers
        mapped_probabilities = np.zeros(len(tokenizer2.vocab))

        for idx, prob in enumerate(probabilities):
            token = tokenizer1.decode([idx])
            new_idx = tokenizer2.encode(token, add_special_tokens=False)[0]
            mapped_probabilities[new_idx] += prob

        # Return the logits of the mapped probabilities
        mapped_logits = np.log(mapped_probabilities)
        return mapped_logits


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

    @classmethod
    def translate_one_to_many(cls, probs_from: torch.FloatTensor, probs_to: torch.FloatTensor,
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

    @classmethod
    def translate_many_to_one(cls, probs_from: torch.FloatTensor, probs_to: torch.FloatTensor,
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

    @classmethod
    def translate_tokenizer_probs(cls, probs: torch.FloatTensor, probs_std: torch.FloatTensor,
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

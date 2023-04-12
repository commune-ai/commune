import commune
import torch
from typing import Dict, Any, List, Tuple
class TokenMapper(commune.Module):
    def __init__(self, from_tokenizer='gpt2', to_tokenizer='facebook/opt-6.7b'):
        self.set_tokenizer_pair(from_tokenizer, to_tokenizer)

    def set_tokenizer_pair(self, from_tokenizer, to_tokenizer):
        
        self.from_tokenizer  = self.get_tokenizer(from_tokenizer)
        self.to_tokenizer  = self.get_tokenizer(to_tokenizer)
        self.translation_map= {
            'to':    self.get_translation_map(self.from_tokenizer, self.to_tokenizer),
            'from':    self.get_translation_map(self.to_tokenizer, self.from_tokenizer)
        }
     
        
        
    def forward(self, logits, from_tokenizer, to_tokenizer):
        logits = self.map_logits_between_tokenizers(logits, self.tokenizers[from_tokenizer], self.tokenizers[to_tokenizer])
        return logits
    
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

if __name__ == "__main__":
    import streamlit as st
    st.write(TokenMapper().__dict__)
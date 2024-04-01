# Readme for file TokenizerMap

The script includes methods to represent the tokenization and detokenization processes that occur in the neural network models, particularly transformers.

The class `TokenizerMap` is used to fetch the tokens (keys to represent words in the model dictionary) from GPT (Generative Pre-training transformers) models. 

It is specially designed for web text analysis but can be expanded for other types of text analysis as well.

## Features:
1. It includes direct token fetching from popular models like GPT-J, GPT-2, GPT-3, gpt-neox etc.
2. It also provides support for translation between tokenizer vocabularies, translating token probability distributions, and aligning tokenized sequences.
3. Allows dynamic addition of new tokenizers.
4. Includes methods for encoding and decoding of top k tokens.

## Main methods:
The class contains the following main methods -

- `set_tokenizer` - Used for initializing the transformer's auto tokenizer from the selected model in the shortcut dictionary.
- `translate_special_token_text` - Translates special tokens (`[BOS][EOS][UNK][SEP][CLS][MASK]`) from one tokenizer to another.
- `set_vocab_len` - Sets the length of the vocabulary if it hasn't been set. 
- `set_whitespace_preserving` - Sets the attribute `whitespace_preserving` in the tokenizer to inform of tokenizers that preserve/don't preserve whitespace.
- `get_translation_map` - Map individual token phrases from a tokenizer to another tokenizer.
- `calculate_loss` - Calculates the cross-entropy loss between predicted and ground truth tokens.
- `tokenize` - Tokenizes the input text according to the selected tokenizer. 
- `token_remap` - Decodes the input token batch and then remaps it using a new tokenizer.

## Usage:
To calculate the cross-entropy loss of a model you can run the `calculate_loss` method. 
For setting a tokenizer you can use the `set_tokenizer` method. 
For tokenizing using the tokenizer selected, you can use the `tokenize` method. 
For translating the special tokens text from one tokenizer to another, use the `translate_special_token_text` method. 
   
## Note:
- The pretrained tokenizers get the sequence of tokens for a given text input. 
- The special tokens used by the transformer models to indicate start/end of a sentence, unknown words, etc. need to be correctly translated when changing tokenizers, as different models may use different tokens for the same purpose. 

## Requirements:
- `transformers`, `torch`, `commune` libraries in python.
- This script works for tokenizers under the `transformers` library. So any updated language model from this library can be used.

# README - Tokenizer Module

The tokenizer module is a Python-based functionality of the Commune library, applied to perform both tokenizing and detokenizing tasks on text data. It simplifies the conversion of text into tokens which can be used as input for language models and converts tokens back into text.

## Key Functions
### Class Initialization:

The class `Tokenizer` is initialized with a default tokenizer. A different tokenizer can be set using the function `set_tokenizer()`.

```python
tokenizer_obj = Tokenizer(tokenizer='gpt2')
```
or
```python
# set tokenizer
tokenizer_obj.set_tokenizer(tokenizer='gpt6b')
```

### Tokenize:

Method `tokenize()` is used to convert a text string into tokens. The padded tokens are returned as torch tensors.

```python
# define text
input_text = "This is an example."
tokenizer_obj.tokenize(text=input_text, max_length=100)
```

### Detokenize:

Method `detokenize()` is used to convert tokens back into text.

```python
# example tensor of tokens
tokens = torch.tensor([4, 32, 78])
tokenizer_obj.detokenize(tokens)
```

### Test:

Class method `test()` is used for testing purposes to print tokenized representation of a string and its detokenized version.

```python
Tokenizer.test()
```
## Usage:

Here are the main steps you should follow when using the Tokenizer module,

1. Import the Commune library and other essential libraries.
2. Instantiate the Tokenizer class.
3. Use the `tokenize()` method to convert your text into tokens or the `detokenize()` method to convert your tokens back into text.

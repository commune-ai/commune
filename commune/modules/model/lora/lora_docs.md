# Model.LoRA

This module involves wrapping over the huggingface transformers library.\
It will allow you to deploy any model in hugginface.

## Training LoRA
We can use pretrained LLMs on huggingface as a base model and train on datasets from huggingface.\
Need specific data preparation function for each dataset.\
\
Example code
```python
from commune.modules.model.lora.lora import LoraModel

adaptor = LoraModel('togethercomputer/LLaMA-2-7B-32K')
adaptor.train('Abirate/english_quotes', 
    './bloke-llama2-7b-abirate-eng-quote-lora-1', data_prep_func)
```

Example of data_prep_func for "Abirate/english_quotes" dataset on huggingface.\
```python
def prep_data(example):
    example['prediction'] = example['quote'] + ' ->: ' + str(example['tags'])
    return example
```
*Supporting causal lms only at the moment.*

## Loading and switching pretrained LoRA
We can load pretrained and locally stored LoRA adaptors.
Empty path will let you use the base model.

Example code
```python
adaptor.load_adaptor('./together-llama2-7b-paper2qa-lora-1')
```

The first time we load adaptors, it will take some time. After initial loading, switching between adaptors will take much less time.
## Generating with LoRA
Once the adaptor is loaded, we can generate some texts using *generate()* method
```python
adaptor.generate('What is this paper about? ->: ')
```

# Model.LoRA

This module involves wrapping over the huggingface transformers library.\
It will allow you to deploy any model in hugginface.

## Training LoRA
We can use pretrained LLMs on huggingface as a base model and train on prepared datasets.\
Need specific data preparation function for each dataset.\
\
Example code
```python
from commune..model.lora.lora import LoraModel

adaptor = LoraModel()
# Init base model on huggingface
adaptor.init_tokenizer('meta-llama/Llama-2-7b-hf')
adaptor.init_base_model('meta-llama/Llama-2-7b-hf')

# Prepare dataset
adaptor.train_data = prepare_data()

# Train the adaptor
adaptor.train()
```

Example of prepare_data function output using "Abirate/english_quotes" dataset on huggingface.\
```text
<s>[INST] Tag the following quote : “Be yourself; everyone else is already taken.” [/INST] ['be-yourself', 'gilbert-perreira', 'honesty', 'inspirational', 'misattributed-oscar-wilde', 'quote-investigator']</s>
```
*Supporting causal llms only at the moment.*

## Loading and switching pretrained LoRA
We can load pretrained and locally stored LoRA adaptors.
Empty path will let you use the base model.

Example code
```python
adaptor.load_adaptor('./mistralai/Mistral-7B-v0.1/eng-quote-tag-lora-32')
```

The first time we load adaptors, it will take some time. After initial loading, switching between adaptors will take much less time.
## Generating with LoRA
Once the adaptor is loaded, we can generate some texts using *generate()* method
```python
adaptor.generate('<s>[INST] Tag the following quote : “Be yourself; everyone else is already taken.” [/INST]')
```

# Huggingface Module

This Python module is designed to facilitate interactions with the Hugging Face models, datasets, and other resources with the help of Hugging Face API through commune and Streamlit libraries. The module also provides simple abstraction over PyTorch and pandas data structures for manipulating model and dataset data.

The module's features include:

- Downloading pretrained models using the Hugging Face hub
- Listing datasets available on the Hugging Face hub with filter options
- Getting tokenizer from given model
- Automatically adjusting shortcuts for known model names
- Viewing and counting unique pipeline tags of models
- Operations on dataframe such as filtering using specific function
- Capturing and visualising snapshot of models
- Manipulating configurations and weights of models
- Running in Streamlit for data visualization with interactive UIs.

## Prerequisites

To use this module, make sure you have the following libraries:

- os, sys
- pandas
- typing
- commune
- streamlit
- torch
- huggingface_hub
- transformers  
- safetensors
- plotly.express
- tokenizers
- accelerate (optional)

## Usage

To utilize the functionalities, you need to instantiate the `Huggingface` class with required configurations. Once the instance is created, you can make use of different methods possessed by Huggingface class like for retrieving the models, getting the tokenizer of a model etc.

```python
huggingface = Huggingface(config)

tokenizer = huggingface.get_tokenizer(model)
model = huggingface.get_model(model_name_or_path)

datasets = huggingface.list_datasets(return_type = 'pandas')
models = huggingface.list_models(return_type = 'pandas')

models_snapshot = huggingface.get_model_snapshots(model)
```

You can run this in Streamlit application by calling the method `streamlit()` of Huggingface instance.

The `class_init()` method should be called to merge functionalities of Huggingface hub and this Huggingface class.

## Tests

The `test()` method can be used to test the functionalities of the Huggingface class.

## Install

Hugging Face can be installed using the `install()` method of Huggingface class or manually installing required dependencies as mentioned in the prerequisites.

## Notes

This is a custom Huggingface wrapper module; for any issues or misinformation, it is recommended to check with the official Hugging Face API documentation. The primary purpose of this module is to provide easier and organized access to the Hugging Face hub models and datasets.
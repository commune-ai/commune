# Gradio example app using model.lora
![Alt text](../imgs/0_interface.png?raw=True "Interface")
## 1. Training
### 1) Model
Select a base model on huggingface. \
***Load Model*** button will download and load the base model.\
Can switch between QLoRA and LoRA in the ***Fine-Tuning Type*** dropdown. QLoRA will load the base model in a quantized type.
Need to specify the ***Adaptor Output Path*** or it will be stored with a tag of ***"lora"*** 

![Alt text](../imgs/1_base_model.png?raw=True "Interface")

### 2) Dataset
![Alt text](../imgs/2_dataset.png?raw=True "Interface")\
Upload dataset files. Currently supporting ***.txt*** file type.
Need to prepare the dataset with some format depending on the base model.
Example of dataset for ***"mistralai/Mistral-7B-v0.1"***.
```text
<s>[INST] Tag the following quote : “Be yourself; everyone else is already taken.” [/INST] ['be-yourself', 'gilbert-perreira', 'honesty', 'inspirational', 'misattributed-oscar-wilde', 'quote-investigator']</s>
```

### 3) Training
![Alt text](../imgs/3_training.png?raw=True "Interface")\
Once the dataset is uploaded, one can train the LoRA using the dataset and selected base model.\
Can config training params.

## 2. Test
![Alt text](../imgs/4_test.png?raw=True "Interface")\
Trained LoRA adaptors are stored on the server and users can switch between them. \
One can load pretrained adaptors and generate texts using them. Need to input the prompt as the same type of dataset used to train the adaptor.
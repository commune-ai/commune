from commune..model.lora.lora import LoraModel
import time, os
from transformers import BitsAndBytesConfig, TrainingArguments
import torch
from datasets import Dataset

######## Example with 'togethercomputer/LLaMA-2-7B-32K' model and 'UNIST-Eunchan/NLP-Paper-to-QA-Generation' dataset ########
stime = time.time()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

adaptor = LoraModel()
adaptor.init_tokenizer('lmsys/vicuna-7b-v1.3')
adaptor.init_base_model('lmsys/vicuna-7b-v1.3', quant_config)
adaptor.training_params = TrainingArguments(
            output_dir='vicuna-eng-quotes',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            optim='paged_adamw_8bit',
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            # group_by_length=True,
            warmup_steps=0,
            lr_scheduler_type='constant',
            remove_unused_columns=False
        )

adaptor.config_lora(
    alpha=32,
    dropout=0.05,
    r=8,
    bias='none',
    task_type='CASUAL_LM'
)
adaptor.init_lora(adaptor.lora_config)


def prep_data(data_path):
    data={'prediction':[]}
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), 'rt', encoding='utf8') as f:
            data['prediction'].append(f.read())

    return Dataset.from_dict(data)

adaptor.train_data = prep_data('/home/v/commune/commune/modules/model/lora/eng_quotes_dataset')

print(f"Base model loading time: {time.time()-stime}")
# def prep_data(example):
#     example['prediction'] = example['question'] + ' ->: ' + example['answer']
#     return example

adaptor.train()

# stime = time.time()
# adaptor.load_adaptor('')
# print(f'LoRA adaptor switching time: {time.time()-stime}')
# adaptor.generate('How does their model learn using mostly raw data? ->: ')
#
# stime = time.time()
# adaptor.load_adaptor('./together-llama2-7b-paper2qa-lora-1')
# print(f'LoRA adaptor initial loading time: {time.time()-stime}')
adaptor.generate('How does their model learn using mostly raw data? ->: ')

######## Example with 'togethercomputer/LLaMA-2-7B-32K' model and 'Abirate/english_quotes' dataset ########
# stime = time.time()
# adaptor = LoraModel('togethercomputer/LLaMA-2-7B-32K')
# print(f"Base model loading time: {time.time()-stime}")
# def prep_data(example):
#     example['prediction'] = example['quote'] + ' ->: ' + str(example['tags'])
#     return example
#
# # adaptor.train('Abirate/english_quotes', './together-llama2-7b-eng-quotes-lora-1', prep_data)
#
# stime = time.time()
# adaptor.load_adaptor('./together-llama2-7b-eng-quotes-lora-1')
# print(f'LoRA switching time: {time.time()-stime}')
# adaptor.generate('Be yourself; everyone else is already taken. ->: ')
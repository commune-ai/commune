from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, \
    pipeline, GenerationConfig, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer
import commune
import torch
import numpy as np

Model = commune.module('model')


class LoraModel(Model):
    def __init__(self, base_model_name):
        self.init_model()
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'

        # Config
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )

        # Model
        self.base_model_ = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=self.quant_config,
            device_map={"": 0}
        )

        self.base_model_.config.use_cache = False
        # self.base_model_.config.pretraining_tp = 1

        # LoRA
        self.lora_config = LoraConfig(lora_alpha=64,
                                      lora_dropout=0.05,
                                      r=32,
                                      bias='none',
                                      task_type='CAUSAL_LM')


    def train(self, dataset_name, output_path):
        # Dataset
        self.train_data = load_dataset(dataset_name, split='train')

        # tokenized_inputs = self.train_data.map(
        #     lambda x: self.tokenizer(x['quote'], truncation=True),
        #     remove_columns=['quote', 'tags', 'author']
        # )
        #
        # input_lengths = [len(x) for x in tokenized_inputs['input_ids']]
        # max_source_length = int(np.percentile(input_lengths, 85))
        #
        # tokenized_targets = self.train_data.map(
        #     lambda x: self.tokenizer(str(x['tags']), truncation=True),
        #     remove_columns=['quote', 'tags', 'author']
        # )
        #
        # target_lengths = [len(x) for x in tokenized_targets['input_ids']]
        #
        # max_target_length = int(np.percentile(target_lengths, 90))

        def prep_data(example):
            # label = self.tokenizer(str(example['tags']), max_length=max_target_length, padding='max_length', truncation=True)
            # label['input_ids'] = [(l if l != self.tokenizer.pad_token_id else -100) for l in label['input_ids']]
            # example = self.tokenizer(example['quote'], max_length=max_source_length, padding='max_length', truncation=True)
            # example['labels'] = label['input_ids']
            # return example

            example['prediction'] = example['question'] + ' ->: ' + str(example['answer'])
            # example['prediction'] = example['quote'] + ' ->: ' + str(example['tags'])
            return self.tokenizer(example['prediction'])

        self.train_data = self.train_data.map(prep_data)
        # self.train_data = self.train_data.map(prep_data, remove_columns=['quote', 'tags', 'author'])
        # self.train_data = self.train_data.map(lambda samples: self.tokenizer(samples), batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        self.train_params = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=1,
            per_device_train_batch_size=4,
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
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type='constant'
        )


        self.lora_model_ = get_peft_model(self.base_model_, self.lora_config)
        self.lora_model_ = prepare_model_for_int8_training(self.lora_model_)
        # self.trainer = Trainer(
        #     model=self.lora_model_,
        #     tokenizer=self.tokenizer,
        #     args=self.train_params,
        #     train_dataset=self.train_data,
        #     data_collator=data_collator
        # )

        self.trainer = SFTTrainer(
            model=self.base_model_,
            train_dataset=self.train_data,
            peft_config=self.lora_config,
            dataset_text_field='prediction',
            tokenizer=self.tokenizer,
            args=self.train_params
        )

        self.trainer.train()
        self.trainer.model.save_pretrained(output_path)

    def load(self, adaptor_path):

        if adaptor_path != '':
            self.lora_model_ = PeftModel.from_pretrained(
                self.base_model_,
                adaptor_path
            )
        elif adaptor_path == '':
            self.lora_model_ = self.base_model_


    def generate(self, prompt):
        self.gen_config = GenerationConfig(
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                num_beams=4,
                no_repeat_ngram_size=3
            )
        inputs = self.tokenizer(prompt, return_tensors='pt').input_ids

        with torch.no_grad():
            gen_output = self.lora_model_.generate(
                input_ids=inputs.cuda(),
                generation_config=self.gen_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )
            s = gen_output.sequences[0]
            self.gen_output = self.tokenizer.decode(s)
            print(self.gen_output)

        # textgen = pipeline(
        #     task='text-generation',
        #     model=adaptor_path,
        #     tokenizer=self.tokenizer,
        #     max_length=200
        # )
        # query = "How do I use the OpenAI API?"
        # output = textgen(f"<s>[INST] {query} [/INST]")
        # print(output[0]['generated_text'])

############# NEW VERSION #############
#
# import os
# import torch
# import torch.nn as nn
# import bitsandbytes as bnb
# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
#
# base_model_name = 'NousResearch/Llama-2-7b-chat-hf'
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     load_in_8bit=True,
#     device_map='auto')
#
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#
# for param in model.parameters():
#     param.requires_grad = False
#     if param.ndim ==1 :
#         param.data = param.data.to(torch.float32)
#
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
#
# class CastOutputToFloat(nn.Sequential):
#     def forward(self, x):
#         return super().forward(x).to(torch.float32)
#
# model.lm_head = CastOutputToFloat(model.lm_head)
#
# from peft import LoraConfig, get_peft_model
# config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
#
# model = get_peft_model(model, config)
#
# # Data
# import transformers
# from datasets import load_dataset
# data = load_dataset("Abirate/english_quotes")
#
# def merge_columns(example):
#     example['prediction'] = example['quote'] + ' ->: ' + str(example['tags'])
#     return example
#
# data['train'] = data['train'].map(merge_columns)
# # data['train']['prediction'][:5]
#
# trainer = transformers.Trainer(model=model,
#                                train_dataset=data['train'],
#                                args=transformers.TrainingArguments(
#                                    per_device_train_batch_size=4,
#                                    gradient_accumulation_steps=4,
#                                    warmup_steps=0,
#                                    max_steps=200,
#                                    learning_rate=2e-4,
#                                    fp16=True,
#                                    logging_steps=1,
#                                    output_dir='outputs'
#                                ),
#                                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
#                                )
# model.config.use_cache = False
# trainer.train()

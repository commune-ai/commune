from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, \
    pipeline, GenerationConfig, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer
import commune
import torch
import gc

Model = commune.module('model')

class LoraModel(Model):
    def __init__(self):
        self.init_model()
        self.tokenizer = None
        self.base_model_ = None
        self.lora_model_ = None
        self.trainer = None
        self.train_data = None
        self.lora_config = None
        self.gen_config = None
        self.use_quant=True

    # def __init__(self, base_model_name):
    #     self.init_model()
    #
    #     # Tokenizer
    #     self.init_tokenizer(base_model_name)
    #
    #     # Config
    #     self.quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type='nf4',
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=False
    #     )
    #
    #     # Model
    #     self.init_base_model(base_model_name=base_model_name, quant_config=self.quant_config)

    def release_memory(self):
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

    def init_tokenizer(self, base_model_name='TheBloke/Llama-2-7b-chat-fp16'):
        try:
            if self.tokenizer:
               self.tokenizer = None
               # self.release_memory()
               try:
                   gc.collect()
                   torch.cuda.empty_cache()
               except:
                   pass

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'
        except:
            self.tokenizer = None
            # self.release_memory()
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            raise ValueError("Base Tokenizer init failure")

    def init_base_model(self, base_model_name='TheBloke/Llama-2-7b-chat-fp16', quant_config=None):
        try:
            self.base_model_name = base_model_name
            if self.base_model_:
                self.base_model_ = None
                # self.release_memory()
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass

            self.base_model_ = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quant_config,
                device_map={"": 0}
            )

            self.base_model_.config.use_cache = False
            self.base_model_.config.pretraining_tp = 1
        except:
            self.base_model_ = None
            self.base_model_name = None
            # self.release_memory()
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            raise ValueError("Base LLM init failure")

    def config_lora(self, alpha=64, dropout=0.05, r=32, bias='none', task_type='CAUSAL_LM'):
        self.lora_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=dropout,
            r=r,
            bias=bias,
            task_type=task_type,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
        )

    def prep_data(self, dataset_name, data_prep_func):
        self.train_data = load_dataset(dataset_name, split='train')

        def prep_data(example):
            example = data_prep_func(example)
            # example['prediction'] = example['question'] + ' ->: ' + str(example['answer'])
            # example['prediction'] = example['quote'] + ' ->: ' + str(example['tags'])
            return self.tokenizer(example['prediction'])

        self.train_data = self.train_data.map(prep_data)


    def init_lora(self, lora_config):
        try:
            self.lora_model_ = get_peft_model(self.base_model_, lora_config)
            self.lora_model_ = prepare_model_for_int8_training(self.lora_model_)
        except:
            self.lora_model_ = None
            # self.release_memory()
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass

            raise ValueError("LoRA init failure")

    def train(self):
        if not self.train_data:
            raise ValueError("Empty training data")

        if not self.training_params:
            raise ValueError("Invalid training parameters")

        if not self.lora_model_:
            raise ValueError("LoRA init failure")

        if not self.base_model_ or not self.tokenizer:
            raise ValueError("Base model init failure")

        # self.prep_data(dataset_name, data_prep_func)
        # # data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        #
        # self.training_params = TrainingArguments(
        #     output_dir=output_path,
        #     num_train_epochs=num_epochs,
        #     per_device_train_batch_size=4,
        #     gradient_accumulation_steps=1,
        #     optim='paged_adamw_8bit',
        #     save_steps=25,
        #     logging_steps=25,
        #     learning_rate=2e-4,
        #     weight_decay=0.001,
        #     fp16=True,
        #     bf16=False,
        #     max_grad_norm=0.3,
        #     max_steps=-1,
        #     warmup_ratio=0.03,
        #     group_by_length=True,
        #     lr_scheduler_type='constant'
        # )
        #
        # self.config_lora(alpha=64, dropout=0.05, r=32, bias='none', task_type='CAUSAL_LM')
        # self.init_lora(self.lora_config)

        self.trainer = SFTTrainer(
            model=self.base_model_,
            train_dataset=self.train_data,
            peft_config=self.lora_config,
            dataset_text_field='prediction',
            tokenizer=self.tokenizer,
            args=self.training_params,
            packing=True
        )

        self.trainer.train()
        self.trainer.model.save_pretrained(self.training_params.output_dir)

    def load_adaptor(self, adaptor_path):
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
        inputs = self.tokenize(prompt).input_ids

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
            return self.gen_output

    def tokenize(self, text: str = 'What\'s up?'):
        return self.tokenizer(text, return_tensors='pt')

    def encode(self, text: str, token_id: int = None, **kwargs) -> torch.Tensor:
        encoded_text = self.tokenize(text)

        return encoded_text

import commune as c
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)

class FineTuner(c.Module):

    def __init__(self, config=None, **kwargs):
        config = self.set_config(config, kwargs=kwargs)
        self.resolve_config()
        self.logger = logging.getLogger(__name__)
        self.set_dataset(config)
        self.set_model(config)
        self.set_trainer(config)
        if config.train:
            self.train()

    def set_model(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        quantization_config = self.get_quantize_config(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, quantization_config=quantization_config)
        if config.load:
            self.load_checkpoint()

    def set_dataset(self, config):

        if c.module_exists(config.dataset.get('module', None)):
            self.dataset = c.module(module)(**config.dataset)
        else:
            self.dataset = load_dataset(**config.dataset)

        sample = self.dataset[0]
        largest_text_field_chars = 0
        if config.trainer.dataset_text_field is None:
            # FIND THE LARGEST TEXT FIELD IN THE DATASET TO USE AS THE TEXT FIELD
            for k, v in sample.items():
                if isinstance(v, str):
                    if len(v) > largest_text_field_chars:
                        largest_text_field_chars = len(v)
                        config.trainer.dataset_text_field = k 
        assert config.trainer.dataset_text_field in sample, f"dataset_text_field {config.trainer.dataset_text_field} not in dataset"

        self.config = config
    quantize_config_map = {'bnb': BitsAndBytesConfig}

    def get_quantize_config(self, config: dict) -> dict:
        '''
        get quantize config
        '''
        if config.quantize.enabled:
            mode = config.quantize.mode

            if mode == 'bnb':
                assert config.quantize.config.bnb_4bit_compute_dtype.startswith('torch.')
                config.quantize.config.bnb_4bit_compute_dtype = eval(config.quantize.config.bnb_4bit_compute_dtype)
            quantization_config = self.quantize_config_map[mode](**config.quantize.config)
        else:
            quantization_config = None
        return quantization_config

    def __call__(self, prompt_text: str, max_length: int = None):
        max_length = max_length if max_length else self.config.max_length
        try:
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise
    
    forward = __call__

    @property
    def checkpoint_path(self) -> str:
        return self.resolve_path( self.config.trainer.args.output_dir + '/' +  self.tag)

    def load_checkpoint(self):
        if c.exists(self.checkpoint_path):
            self.model.from_pretrained(self.checkpoint_path)
        else:
            c.print(f'No checkpoint found at {self.checkpoint_path}')
    
    def save_checkpoint(self):
        if not c.exists(self.checkpoint_path):
            c.mkdir(os.path.dirname(self.checkpoint_path))
        self.trainer.model.save_pretrained(self.checkpoint_path)

    def resolve_config(self) -> str:
        self.config.trainer.args.output_dir = self.model_path
        return config


    def train(self, config):
        self.set_trainer(config)
        self.trainer.train()
        self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint_output_dir = self.config.trainer.output_dir +  "/final_checkpoint"
        self.trainer.model.save_pretrained(checkpoint_output_dir)
    #lora config

    def generate(self, prompt_text: str, max_length: int = None):
        max_length = max_length if max_length else self.config.max_length
        try:
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise

    @classmethod
    def ensure_env(cls):
        c.ensure_libs(['transformers', 'datasets', 'trl', 'peft'])















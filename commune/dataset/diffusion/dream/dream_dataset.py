wimport argparse
import hashlib
import itertools
import math
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import commune
from typing import *

class DreamDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        batch_size:int= 8, 
        tokenizer: Union[str, 'tokenizer'] = None ,
        class_path:str=None,
        # class_prompt:str=None,
        size:int=512,
        center_crop:bool=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.batch_size = batch_size
        self.set_class_path(class_path)
        self.set_tokenizer(tokenizer)


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
            )
    def set_tokenizer(self, tokenizer:Union[str, 'tokenizer'], revision=False, use_fast=False):


        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                                    revision=revision,
                                                    use_fast=use_fast)
        else:
            raise NotImplementedError(type(tokenizer))

        self.tokenizer =tokenizer
        



    def set_class_path(self, class_path:str):

        self.class_path = Path(class_path)
        if not self.class_path.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.class_image_paths = list(self.class_path.iterdir())
        self.num_instance_images = len(self.class_image_paths)
        
        self.class_prompt = os.path.dirname(self.class_path)

        # sentences will be encoded as folders such that "Billys Car"-> billys_car


    def __len__(self):
        return len(self.class_image_paths)


    def __next__(self):
        if not hasattr(self, '_dataloader'):
            self._dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                collate_fn=lambda examples: collate_fn(examples, self.args.with_prior_preservation),
                num_workers=1,
            )

    

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.class_image_paths[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


    @staticmethod
    def collate_fn(examples, with_prior_preservation=False):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch





    @staticmethod
    def parse_args(input_args=None):
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default=None,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            required=False,
            help="Revision of pretrained model identifier from huggingface.co/models.",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            default=None,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--instance_data_dir",
            type=str,
            default=None,
            required=True,
            help="A folder containing the training data of instance images.",
        )
        parser.add_argument(
            "--class_data_dir",
            type=str,
            default=None,
            required=False,
            help="A folder containing the training data of class images.",
        )
        parser.add_argument(
            "--instance_prompt",
            type=str,
            default=None,
            required=True,
            help="The prompt with identifier specifying the instance",
        )
        parser.add_argument(
            "--class_prompt",
            type=str,
            default=None,
            help="The prompt to specify images in the same class as provided instance images.",
        )
        parser.add_argument(
            "--with_prior_preservation",
            default=False,
            action="store_true",
            help="Flag to add prior preservation loss.",
        )
        parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
        parser.add_argument(
            "--num_class_images",
            type=int,
            default=100,
            help=(
                "Minimal class images for prior preservation loss. If there are not enough images already present in"
                " class_data_dir, additional images will be sampled with class_prompt."
            ),
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="text-inversion-model",
            help="The output directory where the model predictions and checkpoints will be written.",
        )
        parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            help=(
                "The resolution for input images, all the images in the train/validation dataset will be resized to this"
                " resolution"
            ),
        )
        parser.add_argument(
            "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
        )
        parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
        parser.add_argument(
            "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
        )
        parser.add_argument(
            "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
        )
        parser.add_argument("--num_train_epochs", type=int, default=1)
        parser.add_argument(
            "--max_train_steps",
            type=int,
            default=None,
            help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        )
        parser.add_argument(
            "--checkpointing_steps",
            type=int,
            default=500,
            help=(
                "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                " training using `--resume_from_checkpoint`."
            ),
        )
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help=(
                "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
            ),
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-6,
            help="Initial learning rate (after the potential warmup period) to use.",
        )
        parser.add_argument(
            "--scale_lr",
            action="store_true",
            default=False,
            help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
        )
        parser.add_argument(
            "--self.lr_scheduler",
            type=str,
            default="constant",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        parser.add_argument(
            "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument(
            "--lr_num_cycles",
            type=int,
            default=1,
            help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
        )
        parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
        parser.add_argument(
            "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
        )
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
        parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
        parser.add_argument(
            "--hub_model_id",
            type=str,
            default=None,
            help="The name of the repository to keep in sync with the local `output_dir`.",
        )
        parser.add_argument(
            "--logging_dir",
            type=str,
            default="logs",
            help=(
                "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
            ),
        )
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default=None,
            choices=["no", "fp16", "bf16"],
            help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            ),
        )
        parser.add_argument(
            "--prior_generation_precision",
            type=str,
            default=None,
            choices=["no", "fp32", "fp16", "bf16"],
            help=(
                "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
            ),
        )
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument(
            "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
        )

        if input_args is not None:
            args = parser.parse_args(input_args)
        else:
            args = parser.parse_args()

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.args.local_rank:
            self.args.local_rank = env_local_rank
        return args


    @classmethod
    def demo(cls):
        import streamlit as st
        class_path = os.path.dirname(__file__) + '/demo_data/python'
        self = cls(class_path=class_path, tokenizer=model)


if __name__ == "__main__":
    DreamDataset.demo()

#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import TEXT_ENCODER_TARGET_MODULES, check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from commune.model.diffusion.dreambooth.dataset import PromptDataset, DreamBoothDataset 
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")
logger = get_logger(__name__)


import commune as c
class Dreambooth(c.Module):
    
    def __init__(self, config=None): 
        self.set_config(config)
        self.set_model(config)
        self.set_dataset(config)
        
        

    # create custom saving & loading hooks so that `self.accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(self, models, weights, output_dir):
        config = self.config
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_lora_layers_to_save = None

        if config.train_text_encoder:
            text_encoder_keys = self.accelerator.unwrap_model(self.text_encoder_lora_layers).state_dict().keys()
        unet_keys = self.accelerator.unwrap_model(self.unet_lora_layers).state_dict().keys()

        for model in models:
            state_dict = model.state_dict()

            if (
                self.text_encoder_lora_layers is not None
                and text_encoder_keys is not None
                and state_dict.keys() == text_encoder_keys
            ):
                # text encoder
                text_encoder_lora_layers_to_save = state_dict
            elif state_dict.keys() == unet_keys:
                # unet
                unet_lora_layers_to_save = state_dict

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        LoraLoaderMixin.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_lora_layers_to_save,
        )

    def load_model_hook(self, models, input_dir):
        # Note we DON'T pass the unet and text encoder here an purpose
        # so that the we don't accidentally override the LoRA layers of
        # unet_lora_layers and text_encoder_lora_layers which are stored in `models`
        # with new torch.nn.Modules / weights. We simply use the pipeline class as
        # an easy way to load the lora checkpoints
        config = self.config
        temp_pipeline = DiffusionPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            revision=config.revision,
            torch_dtype=self.weight_dtype,
        )
        temp_pipeline.load_lora_weights(input_dir)

        # load lora weights into models
        models[0].load_state_dict(AttnProcsLayers(temp_pipeline.unet.attn_processors).state_dict())
        if len(models) > 1:
            models[1].load_state_dict(AttnProcsLayers(temp_pipeline.text_encoder_lora_attn_procs).state_dict())

        # delete temporary pipeline and pop models
        del temp_pipeline
        for _ in range(len(models)):
            models.pop()


    def set_tokenizer(self, config):
        # Load the tokenizer
        if config.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, revision=config.revision, use_fast=False)
        elif config.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                config.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=config.revision,
                use_fast=False,
            )
            
        self.tokenizer = tokenizer


    def prior_preservation(self, config=None):
        config = self.resolve_config(config)
        class_images_dir = Path(config.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
            if config.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif config.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif config.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(config.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=config.sample_batch_size)

            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def set_unet(self, config):
        unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision
        )
        unet.requires_grad_(False)
    
            # Set correct lora layers
        unet_lora_attn_procs = {}
        for name, attn_processor in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = LoRAAttnProcessor

            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        unet.set_attn_processor(unet_lora_attn_procs)
        unet_lora_layers = AttnProcsLayers(unet.attn_processors)



        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(self.accelerator.device, dtype=self.weight_dtype)
        if config.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.unet_lora_layers = unet_lora_layers
        self.unet = unet 
        
        
        
    def set_accelerator(self,config):
        logging_dir = Path(config.output_dir, self.config.logging_dir)

        self.accelerator_project_config = ProjectConfiguration(total_limit=config.checkpoints_total_limit)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=config.report_to,
            logging_dir=logging_dir,
            project_config=self.accelerator_project_config,
        )


        if config.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
        # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if config.train_text_encoder and config.gradient_accumulation_steps > 1 and self.accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )
    
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        
        
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if config.seed is not None:
            set_seed(config.seed)  
        
        
        
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)

            if config.push_to_hub:
                repo_id = create_repo(
                    repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True, token=config.hub_token
                ).repo_id
        
        
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16


    def set_noise_schedular(self, config):
        self.noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")


        
    def set_text_encoder(self,config):
        # import correct text encoder class
        self.set_tokenizer(config)
        text_encoder_cls = self.import_model_class_from_model_name_or_path(config.pretrained_model_name_or_path, config.revision)

        # Load scheduler and models
        text_encoder = text_encoder_cls.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision
        )
        text_encoder.requires_grad_(False)
        text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)


       # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
        # So, instead, we monkey-patch the forward calls of its attention-blocks. For this,
        # we first load a dummy pipeline with the text encoder and then do the monkey-patching.
        text_encoder_lora_layers = None
        if config.train_text_encoder:
            text_lora_attn_procs = {}
            for name, module in text_encoder.named_modules():
                if any(x in name for x in TEXT_ENCODER_TARGET_MODULES):
                    text_lora_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=module.out_features, cross_attention_dim=None
                    )
            text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs)
            temp_pipeline = StableDiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path, text_encoder=text_encoder
            )
            temp_pipeline._modify_text_encoder(text_lora_attn_procs)
            text_encoder = temp_pipeline.text_encoder
            del temp_pipeline


        if config.pre_compute_text_embeddings:

            pre_computed_encoder_hidden_states = self.compute_text_embeddings(config.instance_prompt)
            validation_prompt_negative_prompt_embeds = self.compute_text_embeddings("")

            if config.validation_prompt is not None:
                validation_prompt_encoder_hidden_states = self.compute_text_embeddings(config.validation_prompt)
            else:
                validation_prompt_encoder_hidden_states = None

            if config.instance_prompt is not None:
                pre_computed_instance_prompt_encoder_hidden_states = self.compute_text_embeddings(config.instance_prompt)
            else:
                pre_computed_instance_prompt_encoder_hidden_states = None

            self.text_encoder = None
            self.tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            pre_computed_encoder_hidden_states = None
            validation_prompt_encoder_hidden_states = None
            validation_prompt_negative_prompt_embeds = None
            pre_computed_instance_prompt_encoder_hidden_states = None
            
        self.pre_computed_encoder_hidden_states = pre_computed_encoder_hidden_states
        self.validation_prompt_encoder_hidden_states = validation_prompt_encoder_hidden_states
        self.validation_prompt_negative_prompt_embeds = validation_prompt_negative_prompt_embeds
        self.pre_computed_instance_prompt_encoder_hidden_states = pre_computed_instance_prompt_encoder_hidden_states
        self.text_encoder_lora_layers = text_encoder_lora_layers
        self.text_encoder = text_encoder
        
        




    def encode_prompt(self, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_input_ids = input_ids.to(self.text_encoder.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(self.text_encoder.device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds


    def set_vae(self, config):

        try:
            vae = AutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision
            )
            vae.requires_grad_(False)
        except OSError:
            # IF does not have a VAE so let's just set it to None
            # We don't have to error out here
            vae = None
            
    

        if vae is not None:
            vae.to(self.accelerator.device, dtype=self.weight_dtype)

        self.vae = vae

    def set_model(self, config):
        self.set_accelerator(config)
        self.set_unet(config)
        self.set_text_encoder(config)
        self.set_vae(config)
        self.set_noise_schedular(config)

        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)
        # Generate class images if prior preservation is enabled.
        if config.with_prior_preservation:
            self.prior_preservation(config)
        
        self.set_optimizer(config)



    def resolve_config(self, config):
        return self.config if config == None else config


    def set_dataset(self, config):

        config = self.resolve_config(config)
        # Dataset and DataLoaders creation:
        self.dataset = DreamBoothDataset(
            instance_data_root=config.instance_data_dir,
            instance_prompt=config.instance_prompt,
            class_data_root=config.class_data_dir if config.with_prior_preservation else None,
            class_prompt=config.class_prompt,
            class_num=config.num_class_images,
            tokenizer=self.tokenizer,
            size=config.resolution,
            center_crop=config.center_crop,
            encoder_hidden_states=self.pre_computed_encoder_hidden_states,
            instance_prompt_encoder_hidden_states=self.pre_computed_instance_prompt_encoder_hidden_states,
            tokenizer_max_length=config.tokenizer_max_length,
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: self.collate_fn(examples, config.with_prior_preservation),
            num_workers=config.dataloader_num_workers,
        )



    def set_optimizer(self, config):

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = (
            itertools.chain(self.unet_lora_layers.parameters(), self.text_encoder_lora_layers.parameters())
            if config.train_text_encoder
            else self.unet_lora_layers.parameters()
        )
        
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if config.scale_lr:
            config.learning_rate = (
                config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * self.accelerator.num_processes
            )

        
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

        
        self.lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
            num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
            num_cycles=config.lr_num_cycles,
            power=config.lr_power,
        )




    def train(self, config = None):
        
        config = self.resolve_config(config)
    
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / config.gradient_accumulation_steps)
        
        if config.max_train_steps is None:
            config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        # Prepare everything with our `self.accelerator`.
        if config.train_text_encoder:
            unet_lora_layers, text_encoder_lora_layers, optimizer, self.dataloader, lr_scheduler = self.accelerator.prepare(
                unet_lora_layers, text_encoder_lora_layers, self.optimizer, self.dataloader, self.lr_scheduler
            )
        else:
            unet_lora_layers, optimizer, self.dataloader, lr_scheduler = self.accelerator.prepare(
                unet_lora_layers, optimizer, self.dataloader, lr_scheduler
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / config.gradient_accumulation_steps)
        if overrode_max_train_steps:
            config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth-lora", config=vars(config))

        # Train!
        total_batch_size = config.train_batch_size * self.accelerator.num_processes * config.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.dataloader)}")
        logger.info(f"  Num Epochs = {config.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {config.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if config.resume_from_checkpoint:
            if config.resume_from_checkpoint != "latest":
                path = os.path.basename(config.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(config.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                config.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(config.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * config.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * config.gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, config.num_train_epochs):
            self.unet.train()
            if config.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(self.dataloader):
                # Skip steps until we reach the resumed step
                if config.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % config.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):
                    pixel_values = batch["pixel_values"].to(dtype=self.weight_dtype)

                    if self.vae is not None:
                        # Convert images to latent space
                        model_input = self.vae.encode(pixel_values).latent_dist.sample()
                        model_input = model_input * self.vae.config.scaling_factor
                    else:
                        model_input = pixel_values


                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (model_input.shape[0],), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
                    

                    # Get the text embedding for conditioning
                    if config.pre_compute_text_embeddings:
                        encoder_hidden_states = batch["input_ids"]
                    else:
                        encoder_hidden_states = self.encode_prompt(
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=config.text_encoder_use_attention_mask,
                        )

                    # Predict the noise residual
                    model_pred = self.unet(noisy_model_input, timesteps, encoder_hidden_states).sample

                    # if model predicts variance, throw away the prediction. we will only train on the
                    # simplified training objective. This means that all schedulers using the fine tuned
                    # model must be configured to use one of the fixed variance variance types.
                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    if config.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + config.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet_lora_layers.parameters(), text_encoder_lora_layers.parameters())
                            if config.train_text_encoder
                            else unet_lora_layers.parameters()
                        )
                        self.accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if self.accelerator.is_main_process:
                        if global_step % config.checkpointing_steps == 0:
                            save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break

            if self.accelerator.is_main_process:
                if config.validation_prompt is not None and epoch % config.validation_epochs == 0:
                    logger.info(
                        f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
                        f" {config.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        config.pretrained_model_name_or_path,
                        unet=self.accelerator.unwrap_model(self.unet),
                        text_encoder=None if config.pre_compute_text_embeddings else self.accelerator.unwrap_model(text_encoder),
                        revision=config.revision,
                        torch_dtype=self.weight_dtype,
                    )

                    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                    scheduler_config = {}

                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_config["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, **scheduler_config
                    )

                    pipeline = pipeline.to(self.accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=self.accelerator.device).manual_seed(config.seed) if config.seed else None
                    if config.pre_compute_text_embeddings:
                        pipeline_config = {
                            "prompt_embeds": validation_prompt_encoder_hidden_states,
                            "negative_prompt_embeds": validation_prompt_negative_prompt_embeds,
                        }
                    else:
                        pipeline_config = {"prompt": config.validation_prompt}
                    images = [
                        pipeline(**pipeline_config, generator=generator).images[0] for _ in range(config.num_validation_images)
                    ]

                    for tracker in self.accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {config.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

        # Save the lora layers
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.unet = self.unet.to(torch.float32)
            unet_lora_layers = self.accelerator.unwrap_model(unet_lora_layers)

            if text_encoder is not None:
                self.text_encoder = self.text_encoder.to(torch.float32)
                text_encoder_lora_layers = self.accelerator.unwrap_model(text_encoder_lora_layers)

            LoraLoaderMixin.save_lora_weights(
                save_directory=config.output_dir,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
            )

            # Final inference
            # Load previous pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path, revision=config.revision, torch_dtype=self.weight_dtype
            )

            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_config = {}

            if "variance_type" in pipeline.scheduler.config:
                variance_type = pipeline.scheduler.config.variance_type

                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"

                scheduler_config["variance_type"] = variance_type

            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_config)

            pipeline = pipeline.to(self.accelerator.device)

            # load attention processors
            pipeline.load_lora_weights(config.output_dir)

            # run inference
            images = []
            if config.validation_prompt and config.num_validation_images > 0:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(config.seed) if config.seed else None
                images = [
                    pipeline(config.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    for _ in range(config.num_validation_images)
                ]

                for tracker in self.accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "test": [
                                    wandb.Image(image, caption=f"{i}: {config.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

            if config.push_to_hub:
                self.save_model_card(
                    repo_id,
                    images=images,
                    base_model=config.pretrained_model_name_or_path,
                    train_text_encoder=config.train_text_encoder,
                    prompt=config.instance_prompt,
                    repo_folder=config.output_dir,
                )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=config.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        self.accelerator.end_training()


    @staticmethod
    def collate_fn(examples, with_prior_preservation=False):
        has_attention_mask = "instance_attention_mask" in examples[0]

        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            if has_attention_mask:
                attention_mask += [example["class_attention_mask"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.cat(input_ids, dim=0)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }

        if has_attention_mask:
            batch["attention_mask"] = attention_mask

        return batch



    @staticmethod
    def save_model_card(repo_id: str, images=None, base_model=str, train_text_encoder=False, prompt=str, repo_folder=None):
        img_str = ""
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

        yaml = f"""
        ---
        license: creativeml-openrail-m
        base_model: {base_model}
        instance_prompt: {prompt}
        tags:
        - stable-diffusion
        - stable-diffusion-diffusers
        - text-to-image
        - diffusers
        - lora
        inference: true
        ---
        """
        
        model_card = f"""
        # LoRA DreamBooth - {repo_id}

        These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
        {img_str}

        LoRA for the text encoder was enabled: {train_text_encoder}.
        """
        with open(os.path.join(repo_folder, "README.md"), "w") as f:
            f.write(yaml + model_card)


    @staticmethod
    def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=revision,
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")


    
    def compute_text_embeddings(self,  prompt:str):
        config = self.config
        with torch.no_grad():
            text_inputs = self.tokenize_prompt(prompt, tokenizer_max_length=config.tokenizer_max_length)
            prompt_embeds = self.encode_prompt(
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=config.text_encoder_use_attention_mask,
            )

        return prompt_embeds


    @classmethod
    def test(cls, *args, **kwargs):
        return cls(*args, **kwargs)
        
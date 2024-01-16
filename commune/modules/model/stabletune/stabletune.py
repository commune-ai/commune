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
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""


import subprocess
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import gradio as gr

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from huggingface_hub import model_info
from dotenv import load_dotenv
import commune as c


logger = get_logger(__name__, log_level="INFO")
DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

training_base_model_names = [
    'runwayml/stable-diffusion-v1-5',
    'CompVis/stable-diffusion-v1-4',
    'hakurei/waifu-diffusion-v1-4',
    'stabilityai/stable-diffusion-xl-base-1.0',
]

db_hub_names = [
    "lambdalabs/pokemon-blip-captions",
    "silk-road/MMC4-130k-image-english",
]
# TODO: This function should be removed once training scripts are rewritten in PEFT
def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict

def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
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
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def parse_args_custom(originKeys, _args):
    args = _args
    baseKeys = ["pretrained_model_name_or_path", "revision", "variant", "dataset_name", "dataset_config_name", \
        "train_data_dir", "image_column", "caption_column", "validation_prompt", "num_validation_images", \
        "validation_epochs", "max_train_samples", "output_dir", "cache_dir", "seed", "resolution", "center_crop", \
        "random_flip", "train_batch_size", "num_train_epochs", "max_train_steps", "gradient_accumulation_steps", \
        "gradient_checkpointing", "learning_rate", "scale_lr", "lr_scheduler", "lr_warmup_steps", "snr_gamma", \
        "use_8bit_adam", "allow_tf32", "dataloader_num_workers", "adam_beta1", "adam_beta2", "adam_weight_decay", \
        "adam_epsilon", "max_grad_norm", "push_to_hub", "hub_token", "prediction_type", "hub_model_id", "logging_dir", \
        "mixed_precision", "report_to", "local_rank", "checkpointing_steps", "checkpoints_total_limit", "resume_from_checkpoint", \
        "enable_xformers_memory_efficient_attention", "noise_offset", "rank"]
    
    baseVals = [None, None, None, None, None, None, "image", "text", None, 4, 1, None, "sd-model-finetuned-lora", None, None, 512, \
        False, False, 16, 100, None, 1, False, 1e-4, False, "constant", 500, None, False, False, 0, 0.9, 0.999, 1e-2, 1e-08, 1.0, \
        False, None, None, None, "logs", None, "tensorboard", -1, 500, None, None, False, 0, 4]
    
    numKeys = len(baseKeys)
    for i in range(numKeys):
        if baseKeys[i] not in originKeys:
            setattr(args, baseKeys[i], baseVals[i])
    
    return args

class Stabletune(c.Module):
    def __init__(self):
        load_dotenv()
        self.token = os.getenv("HUGGING_TOKEN")

    def main(self, **kwargs):
        originKeys = kwargs.keys()
        args = argparse.Namespace(**kwargs)
        print(list(originKeys))
        args = parse_args_custom(originKeys, args)
        
        logging_dir = Path(args.output_dir, args.logging_dir)        
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )
        if args.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

            if args.push_to_hub:                
                repo_id = create_repo(
                    # repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
                    args.hub_model_id, exist_ok=True, token=args.hub_token, repo_type='model'
                ).repo_id
                
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        # freeze parameters of models to save more memory
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            # weight_dtype = torch.Tensor.half(torch.float16)
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Freeze the unet parameters before adding adapters
        for param in unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=args.rank, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        unet.add_adapter(unet_lora_config)

        if args.enable_xformers_memory_efficient_attention:
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

        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
                print(data_files)
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image-fine-tune", config=vars(args))

        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, args.num_train_epochs):
            unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective requires that we add one to SNR values before we divide by them.
                            snr = snr + 1
                        mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)                            
                            
                            # unet_lora_state_dict = get_peft_model_state_dict(unet.module)
                            unet_lora_state_dict = get_peft_model_state_dict(unet)                            

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            if accelerator.is_main_process:
                if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device)
                    if args.seed is not None:
                        generator = generator.manual_seed(args.seed)
                    images = []
                    for _ in range(args.num_validation_images):
                        images.append(
                            pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                        )

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unet.to(torch.float32)

            # unet_lora_state_dict = get_peft_model_state_dict(unet.module)
            unet_lora_state_dict = get_peft_model_state_dict(unet)
            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

            if args.push_to_hub:                
                save_model_card(
                    repo_id,
                    images=images,
                    base_model=args.pretrained_model_name_or_path,
                    dataset_name=args.dataset_name,
                    repo_folder=args.output_dir,
                )
                
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                    repo_type='model',
                )
                
        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant, torch_dtype=weight_dtype
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(args.output_dir)

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed)
        images = []
        for _ in range(args.num_validation_images):
            images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if len(images) != 0:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "test": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

        accelerator.end_training()
    def modeltest(self, model_path = None, prompt = "Black Image", output_name = 'new.png'):
        if model_path == None:
            return
        
        info = model_info(model_path)
        model_base = info.cardData['base_model']        

        pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        pipe.unet.load_attn_procs(model_path)
        pipe.to('cuda')

        image = pipe(prompt, num_inference_step=25).images[0]
        image.save(os.path.join("result", output_name))
    def gradio(self):
        with gr.Blocks(title="Stable Diffusion", css="#vertical_center_align_markdown { position:absolute; top:30%;background-color:white;} .white_background {background-color: #ffffff} .none_border {border: none;border-collapse:collapse;}") as demo:            
            with gr.Tab("Text2Image"):                
                with gr.Row():
                    with gr.Column(scale=1, min_width=1):
                        with gr.Group():
                            gr.Markdown("## &nbsp;1. Training")  # , elem_classes="white_background"
                            with gr.Group():
                                gr.Markdown("### &nbsp;1) Model")  # , elem_classes="white_background"
                                with gr.Group():                                            
                                    with gr.Row():  # elem_classes="white_background"
                                        base_model_name = gr.Dropdown(training_base_model_names,
                                                                            label="Base Model Name",
                                                                            value=training_base_model_names[
                                                                                0] if training_base_model_names else None,
                                                                            interactive=True, visible=True, scale=5,
                                                                            allow_custom_value=True)
                                        
                                        
                                    fine_tuning_type_dropdown = gr.Dropdown(["LoRA"],
                                                                            label="Fine-Tuning Type", info="",
                                                                            value="LoRA", interactive=True)
                            with gr.Group():
                                gr.Markdown("### &nbsp;2) Dataset")
                                with gr.Column():   
                                    select_db = gr.Radio(value = 'Huggingface Hub', choices = ['Huggingface Hub', 'Local Drive'], label = 'Select Dataset', interactive=True)
                                    with gr.Row():
                                        db_hub = gr.Dropdown(db_hub_names, label="Huggingface hub datasets",
                                                                            value=db_hub_names[0] if db_hub_names else None,
                                                                            interactive=True, visible=True, scale=1,
                                                                            allow_custom_value=True)
                                        # db_local = gr.Textbox(label='Training dataset directory', interactive=True, visible=True, scale=1)
                                        db_local = gr.Dropdown(sorted(os.listdir("./data")), label="Train Dataset Directory",
                                                                            value=sorted(os.listdir("./data"))[0] if training_base_model_names else None,
                                                                            interactive=True, visible=False, scale=1,
                                                                            allow_custom_value=True)
                            with gr.Group():
                                gr.Markdown("### &nbsp;3) Hub Model ID")
                                with gr.Column():   
                                    hub_model_id = gr.Textbox(
                                        label="Hub Model ID:",
                                        interactive=True,
                                        value="",                                    
                                     )                
                            train_btn = gr.Button(value='Train', interactive=True, scale=1, visible=True)                               
                            train_process = gr.Button(value='Start', interactive=False, scale=1, visible=True)                               
                            
                            
                                                            
                    with gr.Column(scale=1, min_width=1):
                        with gr.Group():
                            gr.Markdown("## &nbsp;2. Test")  # ,elem_classes="white_background"
                            finetuned_model_list = ['Xrunner/test-lora', 'Xrunner/test-victoria', 'Xrunner/test-andy', 'Xrunner/andylau']
                            finetuned_model_list = finetuned_model_list[::-1]
                            finetuned_model = gr.Dropdown(
                                finetuned_model_list,
                                label="Fine-tuned Models",
                                value=finetuned_model_list[0] if finetuned_model_list else None,
                                interactive=True
                            )
                                                        
                            input_prompt_textbox = gr.Textbox(
                                label="Input Prompt:",
                                interactive=True,
                                value="",
                                lines=6,
                                scale=4
                            )
                            image_path_textbox = gr.Textbox(
                                label="Image Path:",
                                interactive=True,
                                value="new.png",                                
                            )
                            with gr.Row():
                                generate_text_btn = gr.Button("Generate", scale=1, interactive=True)
                                
                            output_img = gr.Image(label="Generated Image", type='filepath')
                        
            ### Actions
            def select_db_type(choice):
                print(f"-----------------choice: {choice}----------------------")
                if choice == 'Huggingface Hub':
                    return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
                elif choice == 'Local Drive':
                    return gr.update(visible=False, value=None), gr.update(visible=True, value=None)
            
            def train(base_model, select_db, db_hub, db_local, hub_model_id):
                if select_db == 'Huggingface Hub':
                    MODEL_NAME = base_model
                    DATASET_NAME = db_hub
                    HUB_MODEL_ID = hub_model_id
                    cmd1 = f"huggingface-cli login --token '{self.token}'"
                    print(f"---------------------DATASET_NAME: {DATASET_NAME}------------------")
                    subprocess.call(cmd1, shell=True)
                    cmd2 = f"c model.stabletune main \
                    multi_gpu=True \
                    pretrained_model_name_or_path=$MODEL_NAME \
                    dataset_name='{DATASET_NAME}' \
                    caption_column='text' \
                    resolution=512 \
                    random_flip=True \
                    train_batch_size=4 \
                    num_train_epochs=1 \
                    checkpointing_steps=100 \
                    learning_rate=1e-04 \
                    lr_scheduler='constant' \
                    lr_warmup_steps=0 \
                    seed=42 \
                    output_dir='sd-pokemon-model-lora' \
                    validation_prompt='cute dragon creature' \
                    push_to_hub=True \
                    hub_model_id='{HUB_MODEL_ID}' "
                    cmd = "bash run_hugging.sh"
                    subprocess.call(cmd, shell=True)
                elif select_db == 'Local Drive':                    
                    MODEL_NAME = base_model
                    TRAIN_DIR = db_local
                    HUB_MODEL_ID = hub_model_id
                    
                    cmd1 = f"huggingface-cli login --token '{self.token}'"
                    subprocess.call(cmd1, shell=True)
                    
                    cmd2 = f"c model.stabletune main \
                    multi_gpu=True \
                    pretrained_model_name_or_path='{MODEL_NAME}' \
                    train_data_dir='data/{TRAIN_DIR}' \
                    resolution=512 \
                    random_flip=True \
                    center_crop=True \
                    train_batch_size=4 \
                    checkpointing_steps=50 \
                    learning_rate=1e-04 \
                    gradient_accumulation_steps=4 \
                    validation_prompt='new image' \
                    max_grad_norm=1 \
                    lr_scheduler='constant' \
                    lr_warmup_steps=0 \
                    output_dir='sd-pokemon-model' \
                    hub_model_id='{HUB_MODEL_ID}' \
                    push_to_hub=True \
                    "
                    # cmd = "bash run_custom.sh"
                    subprocess.call(cmd2, shell=True)
                return gr.update(value="completed")
                
            
            def generate(finetuned_model, input_prompt_textbox, image_path_textbox):
                cmd = f"c model.stabletune modeltest '{finetuned_model}' '{input_prompt_textbox}' '{image_path_textbox}'"
                subprocess.call(cmd, shell=True)
                print("-------------------------complete----------------------")
                return gr.update(value=f"result/{image_path_textbox}")
            ### Add functions to component
            select_db.select(select_db_type, inputs = select_db, outputs = [db_hub, db_local])
            train_btn.click(
                train,
                inputs=[base_model_name, select_db, db_hub, db_local, hub_model_id],
                outputs=[train_process],
                queue=False,
            )
            generate_text_btn.click(
                generate,
                inputs=[finetuned_model, input_prompt_textbox, image_path_textbox],
                outputs=[output_img],
                queue=False,
            )
        demo.launch(share=True)
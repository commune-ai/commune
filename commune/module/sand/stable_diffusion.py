import asyncio
import commune as c
from typing import Dict, List
from huggingface_hub import hf_hub_download


class PokemonFineTune(c.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def setup(cls)
        c.run_command(["git", "clone", "https://github.com/justinpinkney/stable-diffusion.git"])
        c.run_command(["pip", "install", "--upgrade", "pip"])
        c.run_command(["pip", "install", "-r", "stable-diffusion/requirements.txt"])
        c.run_command(["pip", "install", "--upgrade", "keras"])
        c.run_command(["pip", "uninstall", "-y", "torchtext"])
        c.run_command(["nvidia-smi"])
        
        
    def fine_tune(self) -> None:

        # Check the dataset
        c.run_command(["pip", "install", "datasets"])
        from datasets import load_dataset

        ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
        sample = ds[0]
        display(sample["image"].resize((256, 256)))
        print(sample["text"])

        # Download weights from Hugging Face Hub
        c.run_command(["pip", "install", "huggingface_hub"])
        from huggingface_hub import notebook_login

        notebook_login()
        ckpt_path = hf_hub_download(
            repo_id="CompVis/stable-diffusion-v-1-4-original",
            filename="sd-v1-4-full-ema.ckpt",
            use_auth_token=True,
        )

        # Set parameters
        BATCH_SIZE = 4
        N_GPUS = 2
        ACCUMULATE_BATCHES = 1
        gpu_list = ",".join((str(x) for x in range(N_GPUS))) + ","
        print(f"Using GPUs: {gpu_list}")

        # Run training
        c.run_command([
            "python", "main.py",
            "-t",
            "--base", "configs/stable-diffusion/pokemon.yaml",
            "--gpus", gpu_list,
            "--scale_lr", "False",
            "--num_nodes", "1",
            "--check_val_every_n_epoch", "10",
            "--finetune_from", ckpt_path,
            "data.params.batch_size=f{BATCH_SIZE}",
            "lightning.trainer.accumulate_grad_batches=f{ACCUMULATE_BATCHES}",
            "data.params.validation.params.n_gpus=f{N_GPUS}",
        ])

        # Run the model
        c.run_command([
            "python", "scripts/txt2img.py",
            "--prompt", "'robotic cat with wings'",
            "--outdir", "'outputs/generated_pokemon'",
            "--H", "512", "--W", "512",
            "--n_samples", "4",
            "--config", "configs/stable-diffusion/pokemon.yaml",
            "--ckpt", "'path/to/your/checkpoint'",
        ])



# Lit-GPT: State-of-the-Art Open-Source Large Language Models

<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

💻 [Website](https://www.lightning.ai/)
📘 [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/) 
🔖 [Fabric Documentation](https://lightning.ai/docs/fabric/stable/)

![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) 
![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg) 
![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT with a pineapple pizza" width="500px"/>

</div>

**Lit-GPT** is a hackable implementation of the most advanced open-source large language models, released under the **Apache 2.0 license**. 

It supports checkpoints from numerous popular public sources, and its design extends on the functionality of [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT). **Lit-GPT** is powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) 🔥.

## Key Features

- **Simplicity:** Single-file implementation without superfluous code.
- **Accuracy:** Ensures numerical equivalence with the original model.
- **Efficiency:** Designed for fast execution on consumer hardware or at scale.
- **Open Source:** No hidden commitments or restrictions.

While our implementation does not prioritize avoiding code duplication, it ensures optimum **readability** and **hackability**.

## Let's Collaborate!

Join our [Discord Channel](https://discord.gg/VptPCZkGNa) and become part of our movement to build high-performance, open-source models for the broader community's benefit.

# Getting Started

## Installation

First, clone the repo:

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Because Lit-GPT relies on PyTorch's nightly Flash Attention, you might need to install it manually until version 2.1 is officially released. Luckily, this is a straightforward process:

With CUDA:

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```

Without CUDA (for CPU-based systems including Macs):

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cpu --pre 
'torch>=2.1.0dev'
```

Optionally, install Flash Attention 2:

```bash
MAX_JOBS=4 pip install 'flash-attn>=2.0.0.post1' --no-build-isolation
```

Finally, install the project dependencies:

```bash
pip install -r requirements.txt
```

Congratulations, you're all set! ⭐

## Using the Model

You need to download the model weights for running text predictions. If you don't have them yet, follow our [guideline](tutorials/download_stablelm.md) for the same. 

You can run inference using:

```bash
python generate/base.py --prompt "Hello, my name is"
```

This command runs the 3B pre-trained model, requiring approximately 7 GB of GPU memory using the `bfloat16` datatype. 

For more detailed information on generating model samples, refer to our [complete guide](tutorials/inference.md).

You can also chat with the model:

```bash
python chat/base.py
```

### Using Big Models on Small Devices

We support 4-bit quantization, similar to QLoRA, LLM.int8, and GPTQ.int4 inference. You can learn more about this in our [quantize guide](tutorials/quantize.md).

## Model Fine-Tuning

We provide several simple training scripts that to finetune pretrained models on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

First, download the dataset and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

Next, run the finetuning script. You can use either:

1. Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):
    
    ```bash
    python finetune/adapter.py
    ```

2. Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):
    
    ```bash
    python finetune/adapter_v2.py
    ```

3. LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

    ```bash
    python finetune/lora.py
    ```

For this, you will need at least one GPU with ~12 GB memory (such as the RTX 3060). Make sure you have also downloaded the pretrained weights as described above. 

For more detailed insight into each of the fine-tuning methods and how to apply them to your own data, do check out our dedicated technical how-to guides.

## Technical Guides for Fine-Tuning

- [Fine-tuning with Adapters](tutorials/finetune_adapter.md)
- [Fine-tuning with LoRA](tutorials/finetune_lora.md)

## Conceptual Tutorials for Fine-Tuning

Not yet familiar with fine-tuning? Check out our additional articles here for a comprehensive introduction:

- [Understanding Parameter-Efficient Fine-tuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)
- [Parameter-Efficient LLM Fine-tuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

## Pre-training

This feature lived in Lit-LLaMA, and we are currently working on porting it into Lit-GPT. Stay tuned for more updates on this!

# Become an Active Contributor 🙌 

It's a thrilling quest as we strive for fully open-source AI. We invite your unique skills and perspectives to refine and expand our mission. We need contributors especially in the following areas:

- [Pre-training Development](https://github.com/Lightning-AI/lit-gpt/labels/pre-training)
- [Fine-tuning Methods](https://github.com/Lightning-AI/lit-gpt/labels/fine-tuning)
- [Quantization Techniques](https://github.com/Lightning-AI/lit-gpt/labels/quantization)
- [Sparsification Procedures](https://github.com/Lightning-AI/lit-gpt/labels/sparsification)

Join us regardless of your experience level or hardware strength. Your contribution matters – check out our simplified [Contribution Guide](https://lightning.ai/pages/community/tutorial/contributing-to-lit-llama-a-hitchhikers-guide-to-the-quest-for-fully-open-source-ai/), meant to facilitate your involvement with the Lit-GPT project.

Also, don't forget to [join our Discord community](https://discord.gg/VptPCZkGNa)!

## Acknowledgements

We wish to express our gratitude to the following developers and their projects:

- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@EleutherAI](https://github.com/EleutherAI) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@tridao](https://github.com/tridao) for [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)

# License

Lit-GPT is made available under the [Apache 2.0 license](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE).
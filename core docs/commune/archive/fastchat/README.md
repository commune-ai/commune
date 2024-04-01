# FastChat: Training and Evaluating Language Model-Based Chatbots

FastChat provides an open platform for training, serving, and evaluating large scale language model chatbots. 

## Latest Release

<p align="center">
<a href="https://vicuna.lmsys.org"><img src="assets/vicuna_logo.jpeg" width="20%"></a>
</p>

We have recently launched **Vicuna: An Open-Source Chatbot boasting GPT-4 with 90% ChatGPT performance**. Get insights from our latest blog [post](https://vicuna.lmsys.org) and try out the [demo](https://chat.lmsys.org/).

<a href="https://chat.lmsys.org"><img src="assets/demo_narrow.gif" width="70%"></a>

Stay up-to-date by joining our [Discord](https://discord.gg/h6kCZb72G7) server and following us on [Twitter](https://twitter.com/lmsysorg).

## Overview
- [Installation Instructions](#install)
- [Tutorial on Vicuna Weights](#vicuna-weights)
- [How to Run Inference with Command Line Interface](#inference-with-command-line-interface)
- [Guide on Serving with Web GUI](#serving-with-web-gui)
- [API Information](#api)
- [Evaluation Procedure](#evaluation)
- [Fine-Tuning Details](#fine-tuning)

## How to Install

### Option 1: Using pip

```bash
# FastChat Installation
pip3 install fschat

# Install the latest branch of huggingface/transformers
pip3 install git+https://github.com/huggingface/transformers
```

### Option 2: From Source Code

1. Clone this repository and navigate to the FastChat directory
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

If you are running on a Mac:
```bash
brew install rust cmake
```

2. Install the Package
```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .
```

## Vicuna Weights

Vicuna weights are released as delta weights, in line with the LLaMA model license.
Follow these instructions to get Vicuna weights:

1. Obtain original LLaMA weights using huggingface format from [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use scripts to obtain Vicuna weights by applying our delta. Delta weights can be automatically downloaded from our Hugging Face [account](https://huggingface.co/lmsys).

**Note**:
Our weights are only compatible with the latest huggingface/transformers branch.
Fastchat ensures the correct transformers version during installation.

## Running with Command Line Interface

(Experimental: You can enable rich text output and improve text streaming quality for non-ASCII content with `--style rich`. This may not work properly on certain terminals.)

#### Single GPU
The command below necessitates around 28GB of GPU memory for Vicuna-13B and 14GB for Vicuna-7B.
To alleviate memory constraints, consider the section "Insufficient Memory or Other Platforms".

#### Multiple GPUs
Aggregate GPU memory through model parallelism on the same machine.

#### CPU Only
Operating on a CPU only doesn't require a GPU. It requires around 60GB of CPU memory for Vicuna-13B and about 30GB for Vicuna-7B.

#### Metal Backend (Apple Silicon Macs or AMD GPUs)
Enabling GPU acceleration on Macs requires torch ≥ 2.0. Turn on 8-bit compression with `--load-8bit`.

#### Insufficient Memory or Other Platforms
Enable 8-bit compression to reduce memory usage by half with slight degradation in model quality. This is compatible with CPU, GPU, and Metal backend. Contributions to further improve the model's functionality are welcome.

## Serving with Web GUI

To serve models using the web UI, start by launching the controller that manages the distributed workers, then launch the model worker, and finally the Gradio web server.

## API

### Huggingface Generation APIs
See [fastchat/serve/huggingface_api.py](fastchat/serve/huggingface_api.py)

### RESTful APIs Compatible with OpenAI
Coming soon.

## Evaluation 

Our evaluation pipeline uses GPT-4. It consists of generating answers from different models, creating reviews with GPT-4, and visualizing the data.

## Fine-tuning 

Vicuna is fine-tuned using approximately 70K user-shared conversations from ShareGPT.com with public APIs. We may not release the ShareGPT dataset currently. You can perform fine-tuning using some dummy questions in [dummy.json](playground/data/dummy.json) in the same format as your own data. 

Our fine-tuning procedure is based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), with additional support for multi-round conversations. Training Vicuna-7B with Local GPUs requires certain memory requirements. Vicuna can be trained on any cloud platform with [SkyPilot](https://github.com/skypilot-org/skypilot), a UC Berkeley framework for running ML workloads cost-effectively.
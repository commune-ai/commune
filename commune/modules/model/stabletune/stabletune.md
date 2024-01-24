# Stable Diffusion text-to-image fine-tuning

The module shows how to fine-tune stable diffusion model on your own dataset.

## Running locally with PyTorch
### Installing the dependencies

Before running the script, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

### Pokemon example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4` and `v1-5`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree.

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

```bash
huggingface-cli login --token your_own_token
```

<br>

#### Hardware
With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB GPU. For higher `batch_size` and faster training it's better to use GPUs with >30GB memory.

## Training

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

With LoRA, it's possible to fine-tune Stable Diffusion on a custom image-caption pair dataset
on consumer GPUs like Tesla T4, Tesla V100.

### Training with Huggingface dataset

First, you need to set up your development environment as is explained in the [installation section](#installing-the-dependencies). Make sure to set the `MODEL_NAME` and `DATASET_NAME` environment variables. Here, we will use [Stable Diffusion v1-4](https://hf.co/CompVis/stable-diffusion-v1-4) and the [Pokemons dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions).

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [Weights and Biases](https://docs.wandb.ai/quickstart) is a nice solution to easily see generating images during training. All you need to do is to run `pip install wandb` before training to automatically log images.___**

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export HUB_MODEL_ID="test-lora"

```

For this example we want to directly store the trained LoRA embeddings on the Hub, so
we need to be logged in and add the `--push_to_hub` flag.

```bash
huggingface-cli login --token Your_Own_Huggingface_Repo_Token
```

Now we can start training!

```bash
c model.stabletune main \
  multi_gpu=True \
  pretrained_model_name_or_path=$MODEL_NAME \
  dataset_name=$DATASET_NAME \
  caption_column="text" \
  resolution=512 \
  random_flip=True \
  train_batch_size=4 \
  num_train_epochs=1 \
  checkpointing_steps=100 \
  learning_rate=1e-04 \
  lr_scheduler="constant" \
  lr_warmup_steps=0 \
  seed=42 \
  output_dir="sd-pokemon-model-lora" \
  validation_prompt="cute dragon creature" \
  push_to_hub=True \
  hub_model_id=${HUB_MODEL_ID} \  
```
Simpley run the following command.
```bash
bash run.sh
```
The above command will also run inference as fine-tuning progresses and log the results to Weights and Biases.

**___Note: When using LoRA we can use a much higher learning rate compared to non-LoRA fine-tuning. Here we use *1e-4* instead of the usual *1e-5*. Also, by using LoRA, it's possible to run `stabletune` in consumer GPUs like T4 or V100.___**

The final LoRA embedding weights have been uploaded to [sayakpaul/sd-model-finetuned-lora-t4](https://huggingface.co/sayakpaul/sd-model-finetuned-lora-t4). **___Note: [The final weights](https://huggingface.co/sayakpaul/sd-model-finetuned-lora-t4/blob/main/pytorch_lora_weights.bin) are only 3 MB in size, which is orders of magnitudes smaller than the original model.___**

You can check some inference samples that were logged during the course of the fine-tuning process [here](https://wandb.ai/sayakpaul/text2image-fine-tune/runs/q4lc0xsw).

### Training with custom dataset

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**

**___Note: It is quite useful to monitor the training progress by regularly generating sample images during training. [Weights and Biases](https://docs.wandb.ai/quickstart) is a nice solution to easily see generating images during training. All you need to do is to run `pip install wandb` before training to automatically log images.___**

### Prepare Dataset
./data/folder_name/

There are image files and metadata.jsonl.

metadata.jsonl example:

```Jsonl
{"file_name": "14.jpg", "text": "Andy Lau in a suit and bow tie making a peace sign with his hands and a watch on his wrist"}

```


You need at least 15 training images.

It is okay to have images with different aspect ratios. Make sure to turn on the bucketing option in training, which sorts the images into different aspect ratios during training.

Pick images that are at least 512×512 pixels for v1 models.

Make sure the images are either PNG or JPEG formats.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="data/cat"
export HUB_MODEL_ID="test-lora"
```

For this example we want to directly store the trained LoRA embeddings on the Hub, so
we need to be logged in and add the `--push_to_hub` flag.

```bash
huggingface-cli login --token Your_Own_Huggingface_Repo_Token
```

Now we can start training!

```bash
c model.stabletune main \
  multi_gpu=True \
  pretrained_model_name_or_path=$MODEL_NAME \
  train_data_dir=$TRAIN_DIR \
  resolution=512 \
  random_flip=True \
  center_crop=True \
  train_batch_size=4 \
  checkpointing_steps=100 \
  learning_rate=1e-04 \
  gradient_accumulation_steps=4 \
  validation_prompt="cute cat creature" \
  max_grad_norm=1 \
  lr_scheduler="constant" \
  lr_warmup_steps=0 \
  output_dir="sd-pokemon-model" \
  hub_model_id=${HUB_MODEL_ID} \
```

The above command will also run inference as fine-tuning progresses and log the results to Weights and Biases.

Simpley run the following command.
```bash
bash run_hugging.sh
```

### Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline` after loading the trained LoRA weights.  You
need to pass the `output_dir` for loading the LoRA weights which, in this case, is `sd-pokemon-model-lora`.

```bash
c model.stabletune modeltest Model_Path
c model.stabletune modeltest Model_Path prompt output_path
```


### Gradio:

```
c model.stabletune gradio
```


```python
finetuned_model_list = ['Hub User-1/Hub Name1', 'Hub User-1/Hub Name2', ...]
```
Example: finetuned_model_list = ['Xrun/test-cat', 'Xrun/beckham', ...]

```python
with gr.Column(scale=1, min_width=1):
    with gr.Group():
        gr.Markdown("## &nbsp;2. Test")  # ,elem_classes="white_background"
        finetuned_model_list = ['Xrun/test-lora', 'Xrun/test-victoria', 'Xrun/test-andy', 'Xrun/andylau']
        finetuned_model_list = finetuned_model_list[::-1]
        finetuned_model = gr.Dropdown(
            finetuned_model_list,
            label="Fine-tuned Models",
            value=finetuned_model_list[0] if finetuned_model_list else None,
            interactive=True
        )
```
You can replace 'Xrun', 'test-lora', 'test-victoria', 'test-andy', and 'andylau' with yours.


Generated image will be saved in the "result' folder of the current directory.
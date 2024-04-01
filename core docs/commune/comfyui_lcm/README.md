# Latent Consistency Model (LCM) for ComfyUI <!-- omit from toc -->

ComfyUI-LCM is a focused extension that seamlessly integrates LCMs, a unique class of models, into the user-friendly ComfyUI interface. It capitalizes on the diffusers library, unlike the base Comfy that makes use of its inbuilt model loading mechanism. The principal highlight of this extension is the LCM_Dreamshaper_v7 checkpoint's implementation. 

## Installation <!-- omit from toc -->

You can easily install the extension by cloning this repository into your `custom_nodes/` directory before restarting ComfyUI:
```
git clone https://github.com/0xbitches/ComfyUI-LCM
```

## Features <!-- omit from toc -->

The LCM extension arrives with the basic img2img functionality through the `LCM_img2img_Sampler` node. 

For advanced features such as vid2vid, please consider installing the ComfyUI-VideoHelperSuite helper node. To create a full vid2vid workflow, use the Load Video and Video Combine nodes. 

## Workflows <!-- omit from toc -->

For ease of use, LCM extension also includes ready-made workflow options. These encompass simple and advanced variants for txt2img, img2img, and vid2vid that can be effortlessly downloaded or dragged to Comfy.

## Issue Resolution <!-- omit from toc -->

Despite our relentless drive for perfection, you might encounter the known issue involving ValueError for a non-consecutive added token. But worry not! We always have workarounds and quick-fixes detailed in our workflow instructions. 

## Workflows

To use these workflows, download or drag the image to Comfy.

### LCM txt2img simple

![txt2img](./assets/lcm_txt2img.png)

### LCM img2img simple

![img2img](./assets/lcm_img2img.png)

### LCM vid2vid simple

![img2img](./assets/lcm_vid2vid.png)

### LCM txt2img advanced

![img2img](./assets/lcm_txt2img_advanced.png)

### LCM img2img advanced

![img2img](./assets/lcm_img2img_advanced.png)

### LCM vid2vid advanced

![img2img](./assets/lcm_vid2vid_advanced.png)

## Known Issues

#### `ValueError: Non-consecutive added token '<|startoftext|>' found. Should have index 49408 but has index 49406 in saved vocabulary.`

To resolve this, locate your huggingface hub cache directory.

It will be something like `~/.cache/huggingface/hub/path_to_lcm_dreamshaper_v7/tokenizer/`. On Windows, it will roughly be `C:\Users\YourUserName\.cache\huggingface\hub\models--SimianLuo--LCM_Dreamshaper_v7\snapshots\c7f9b672c65a664af57d1de926819fd79cb26eb8\tokenizer\`.

Find the file `added_tokens.json` and change the contents to:

```
{
  "<|endoftext|>": 49409,
  "<|startoftext|>": 49408
}
```

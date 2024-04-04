# Audiocraft - Transformative Audio Generation

![docs badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/facebookresearch/audiocraft/workflows/audiocraft_tests/badge.svg)

Audiocraft is a revolutionary PyTorch library designed to support deep learning research in the domain of audio generation. The major highlight of Audiocraft is its ability to generate music through MusicGen, a top-notch controllable text-to-music model.

## A Glimpse into MusicGen

MusicGen, powered by Audiocraft, is a groundbreaking model for simple and controllable music generation. This single-stage auto-regressive Transformer model is trained over a 32kHz EnCodec tokenizer using 4 codebooks sampled at 50 Hz. In comparison to existing models like MusicLM, MusicGen eliminates the need for a self-supervised semantic representation, generating all four codebooks in one pass. With a subtle delay introduced between codebooks, they can be predicted in parallel, resulting in only 50 auto-regressive steps for each second of audio. Experience MusicGen through our [sample page][musicgen_samples] or test the demo available!

<a target="_blank" href="https://colab.research.google.com/drive/1-Xe9NCdIs2sCUbiSmwHXozK6AAhMm7_i?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/facebook/MusicGen">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>

To provide optimal results, MusicGen is trained utilizing 20K hours of licensed music. This incorporates a proprietary dataset of 10,000 premium-quality music tracks and data from ShutterStock and Pond5 music.

## Getting Started

For Audiocraft to function properly, your system needs to have Python 3.9, PyTorch 2.0.0, and a GPU with a minimum of 16 GB of memory. The medium-sized model demands these resources. Installing Audiocraft is as simple as running the following commands:

```shell
pip install 'torch>=2.0'
pip install -U audiocraft  # for stable release
pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # for the latest version
pip install -e .  # in case you cloned the repository locally
```

## How to Use 

Interacting with MusicGen is feasible in various ways:

1. An interactive demo is available at the [`facebook/MusicGen` HuggingFace Space](https://huggingface.co/spaces/facebook/MusicGen).
2. Run the extended demo via this [Colab notebook](https://colab.research.google.com/drive/1fxGqfg96RBUvGxZ1XXN07s3DthrKUl4-?usp=sharing).
3. Implement the gradio demo locally by initiating `python app.py`.
4. By running [`demo.ipynb`](./demo.ipynb) on your local Jupyter notebook.
5. Visit [@camenduru's Colab page](https://github.com/camenduru/MusicGen-colab) for noteworthy community contributions.

## Deploying the API

We provide four pre-trained models through a user-friendly API. The models include `small`, `medium`, `melody`, and `large`. For optimum balance between quality and computation, we recommend using the `medium` or `melody` model. If you're using MusicGen locally, remember that **a GPU is a mandatory requirement**.

**Note:** Newer `torchaudio` versions necessitate installing [ffmpeg](https://ffmpeg.org/download.html) as well:

```
apt-get install ffmpeg
```

Below is a simple use-case for the API:

```python
...
```

## Exploring the Model Card

For further details, check out [the model card page](./MODEL_CARD.md).

## Addressing Common Queries

#### Training Code Release

The training code for both MusicGen and EnCodec will be revealed soon.

#### Assistance for Windows Users

@FurkanGozukara provides a comprehensive tutorial for [Audiocraft/MusicGen on Windows](https://youtu.be/v-YpvPkhdO4).

#### Demo Assistance for Colab 

For any issues in running the demo on Colab, refer to [@camenduru's tutorial on YouTube](https://www.youtube.com/watch?v=EGfxuTy9Eeo).

## Citing Audiocraft

If you are leveraging Audiocraft in your research, please cite our work as follows:

```
...
```

## Licensing Details

* The code in this repository is licensed under the MIT license as detailed in the [LICENSE file](LICENSE).
* The weights in this repository are covered by the CC-BY-NC 4.0 license as detailed in the [LICENSE_weights file](LICENSE_weights).
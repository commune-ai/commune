# Musicgen

This is the music generation module in the Commune.
Audiocraft is a PyTorch library for deep learning research on audio generation.
Audiocraft provides the code and models for MusicGen. MusicGen is a single stage auto-regressive
Transformer model trained over a 32kHz <a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with a codebook sampled at 50 Hz. Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require a self-supervised semantic representation.

## Installation
Audiocraft requires Python 3.10, PyTorch 2.0.0, and a GPU with at least 16 GB of memory (for the medium-sized model). To install Audiocraft, you can run the following:

```shell
pip install 'torch>=2.0'
pip install -U audiocraft
```

**Note**: Please make sure to have [ffmpeg](https://ffmpeg.org/download.html) installed when using newer version of `torchaudio`.
You can install it with:
```
apt get install ffmpeg
```

## Usage
`c model.musicgen gradio`

## More details
The model will generate a short music extract based on the description you provided.
You can generate up to 30 seconds of audio.
When using `melody`, ou can optionaly provide a reference audio from
which a broad melody will be extracted. The model will then try to follow both the description and melody provided.
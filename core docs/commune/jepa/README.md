# V-JEPA: An Advanced Method for Learning Visual Representations from Video

Welcome to the official PyTorch codebase for the innovative video joint-embedding predictive architecture, V-JEPA. This is an advanced method for self-supervised learning of visual representations from video. Developed with research support from [Meta AI Research, FAIR](https://ai.facebook.com/research/). 

Authors: Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas.

[Check out our Blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
[Read our Research Paper](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)

V-JEPA models are expertly trained using the VideoMix2M dataset and can create versatile visual representations that excel at downstream video and image tasks, without necessitating any adjustment in the model’s parameters.

## How V-JEPA Works
V-JEPA pretraining is grounded on an entirely unsupervised feature prediction objective, and does not rely on pretrained image encoders, text, negative examples, human annotations, or pixel-level reconstruction. 

Visualizations: V-JEPA encompasses a predictor that makes its predictions in the latent space, unlike generative methods that have a pixel decoder. For the visualization of interpretations being represented in pixels, the predictions made in V-JEPA's feature space are decoded using a trained conditional diffusion model. Here, the unmasked regions of the video do not have access to the decoder, and the decoder is only provided with the representations predicted for the video's missing regions. 

## Model Collections

All models for pre-training and attentive probes for evaluation are provided and readily available for download. 

## Repository Overview 

The repository has a structured framework, including applications for training loops, evaluations for 'apps', and config files specifying experiment parameters for app/eval runs. 

## Preparing the Data

V-JEPA pretraining and evaluations are compatible with many standard video formats. For easy integration, create a ```.csv``` file using our laid out format. Image datasets use the PyTorch ```ImageFolder``` class. 

## Execution

Both local and distributed training are available for V-JEPA pretraining and evaluations. For local training, the configuration files are parsed, and pretraining is performed locally on a multi-GPU or single-GPU machine. For distributed training, the configuration file is parsed, and the training details are specified. 

## Setting Up

Create a new anaconda environment, activate it, and run the setup.py script.

## License

[Here](./LICENSE) is our licensing details. 

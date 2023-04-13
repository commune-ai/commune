from safetensors.numpy import save_file, load_file
import numpy as np
import commune
import streamlit as st


hf = commune.get_module('huggingface')()
config = hf.get_model_weights('vicuna-13b-GPTQ-4bit-128g')
import os
commune.print(config)

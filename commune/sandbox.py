import commune
import streamlit as st
import torch

# commune.get_module('dataset.text.huggingface').deploy_fleet()
commune.get_module('model.transformer').deploy_fleet()

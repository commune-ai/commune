import os
import argparse
from pathlib import Path

import torch
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

CKPT_PRETRAINED = Path("/ckpt/pretrained")


model_name = 'EleutherAI/gpt-neox-20b'
weights_path = f"/ckpt/pretrained/{model_name}"
config = AutoConfig.from_pretrained(model_name)

config.use_cache = False

with init_empty_weights():
  model = AutoModelForCausalLM.from_config(config)

tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

device_map = infer_auto_device_map(
    model, no_split_module_classes=["GPTNeoXLayer"],dtype=torch.bfloat16, #note: succeeds with float16 as well.
    max_memory = {0: "15GiB", 1: "15GiB",3: "15GiB" , 'cpu': "20GiB"},
    )

weights_path = model_name

device_map['gpt_neox.embed_in'] = 'cpu'
print(f"device_map: {device_map}")
load_checkpoint_and_dispatch(
    model,
    weights_path,
    device_map=device_map,
    offload_folder=None,
    offload_state_dict=False,
    dtype="bfloat16"
  )

print(model)

model = model.eval()
prompt = "Deepspeed is "
m_inp = tokenizer(prompt, return_tensors="pt")
attn_mask = m_inp.get("attention_mask", None).to(device='cuda:0')

with torch.no_grad():
  gen_tokens = model.generate(
    m_inp["input_ids"].to(0), attention_mask = attn_mask,
    do_sample=True, max_new_tokens=100, temperature=0.9
  )
gen_text = tokenizer.decode(output[0].tolist())
print(f"generated tokens: {gen_text}")
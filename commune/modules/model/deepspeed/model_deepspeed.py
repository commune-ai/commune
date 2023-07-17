import commune as c
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
import time
from deepspeed.runtime.utils import see_memory_usage
'''
Helper classes and functions for examples
'''

import os
import io
from pathlib import Path
import json
import deepspeed
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast


class DeepSpeed(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config=config, kwargs=kwargs)

        config = self.config
        c.print(config)
        config.model = config.shortcuts.get(config.model, config.model)
        if not config.ds_inference and config.world_size > 1:
            raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

        data_type = getattr(torch, config.dtype)

        if config.local_rank == 0:
            see_memory_usage("before init", True)

        t0 = time.time()

        self.dtype = data_type


        # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, padding_side=config.padding_side)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if (is_meta):
            '''When meta tensors enabled, use checkpoints'''
            self.config = AutoConfig.from_pretrained(config.model)

            self.repo_root, self.checkpoints_json = self._generate_json(config.checkpoint_path)

            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model,  trust_remote_code=True)

        self.model.eval()

        if self.dtype == torch.float16:
            self.model.half()
        if config.local_rank == 0:
            print(f"initialization time: {(time.time()-t0) * 1000}ms")
            see_memory_usage("after init", True)
        if config.use_meta_tensor:
            ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
        else:
            ds_kwargs = dict()

        if config.ds_inference:
            self.model = deepspeed.init_inference(
                self.model,
                dtype=self.data_type,
                mp_size=config.world_size,
                replace_with_kernel_inject=config.use_kernel,
                replace_method=config.replace_method,
                max_tokens=config.max_tokens,
                save_mp_checkpoint_path=config.save_mp_checkpoint_path,
                
                **ds_kwargs
            )
        if config.local_rank == 0:
            see_memory_usage("after init_inference", True)

        self.config = config

    
    
    
    def test(self):
        input_sentences = [
            "DeepSpeed is a machine learning framework",
            "He is working on",
            "He has a",
            "He got all",
            "Everyone is happy and I can",
            "The new movie that got Oscar this year",
            "In the far far distance from our galaxy,",
            "Peace is the only way"
        ]

        if config.batch_size > len(input_sentences):
            # dynamically extend to support larger bs by repetition
            input_sentences *= math.ceil(config.batch_size / len(input_sentences))

        inputs = input_sentences[:config.batch_size]
        iters = config.test.iters  # warmup

        times = []
        for i in range(iters):
            response = self.forward(inputs, num_tokens=config.max_new_tokens, do_sample=(not config.greedy))
            times.append(response['latency'])
        print(f"generation time is {times[-1]} sec")
        if config.local_rank == 0:
            for i, o in zip(inputs, outputs):
                print(f"\nin={i}\nout={o}\n{'-'*60}")
            self.print_perf_stats(map(lambda t: t / config.max_new_tokens, times), pipe.model.config)

    def forward(self,
                inputs=["test"],
                num_tokens=100,
                do_sample=False):
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        outputs = self.generate_outputs(input_list, num_tokens=num_tokens, do_sample=do_sample)
        return outputs

        if num_tokens is None:
            num_tokens = self.config.max_new_tokens

        torch.cuda.synchronize()
        start = time.time()
        outputs = self.pipe(inputs, num_tokens=num_tokens, do_sample=do_sample, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        latency = end - start

        return {
            'outputs': outputs,
            'latency': latency
        }



    def _generate_json(self, checkpoint_path=None):
        if checkpoint_path is None:
            repo_root = snapshot_download(self.config.model,
                                      allow_patterns=["*"],
                                      cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                                      ignore_patterns=["*.safetensors"],
                                      local_files_only=False,
                                      revision=None)
        else:
            assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
            repo_root = checkpoint_path

        if os.path.exists(os.path.join(repo_root, "ds_inference_config.json")):
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        elif (config.model in config.tp_presharded_models):
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            checkpoints_json = "checkpoints.json"

            with io.open(checkpoints_json, "w", encoding="utf-8") as f:
                file_list = [str(entry).split('/')[-1] for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
                data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
                json.dump(data, f)

        return repo_root, checkpoints_json


    def generate_outputs(self,
                         inputs=["test"],
                         num_tokens=100,
                         do_sample=False):
        generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=do_sample)

        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)

        self.model.cuda().to(self.device)

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            # NOTE: Check if Llamma can work w/ **input_tokens
            #       'token_type_ids' kwarg not recognized in Llamma generate function
            outputs = self.model.generate(input_tokens.input_ids, **generate_kwargs)
        else:
            outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs


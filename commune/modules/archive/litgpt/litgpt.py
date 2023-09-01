import commune as c
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Literal


import torch


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


class LitGpt(c.Module):
    def __init__(self,config= None, **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)
        torch.set_float32_matmul_precision(config.float32_matmul_precision)
        self.set_model(model=config.model,
                        quantize=config.quantize, 
                        strategy=config.strategy, 
                        devices=config.devices, 
                        precision=config.precision,
                        seed = config.seed,
                        test = config.test
                        )
        if config.test:
            self.test(model=self)
        
    

    def set_model(self,
        model: str = 'vicuna.13b',
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
        strategy: str = "auto",
        devices: int = None,
        precision: str = "bf16-true",
        seed : bool = 42,
        test: bool = False,
    ) -> None:
        """Generates text samples based on a pre-trained model and tokenizer.

        Args:
            model: The model
            quantize: Whether to quantize the model and using which method:
                - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
                - bnb.int8: 8-bit quantization from bitsandbytes
                - gptq.int4: 4-bit quantization from GPTQ
                for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
            strategy: Indicates the Fabric strategy setting to use.
            devices: How many devices to use.
            precision: Indicates the Fabric precision setting to use.
        """
        from lightning.fabric.strategies import FSDPStrategy
        import lightning as L
        from lit_gpt import GPT, Tokenizer, Config
        from lit_gpt.model import Block
        from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, quantization


        model = self.resolve_model(model) # lets get the model boys
        if devices is None:
            devices = c.model_max_gpus(model)

        c.print(f'Using {devices} devices for model {model}', color='green')
        checkpoint_dir = Path(self.get_model_checkpoint(model))
        if strategy == "fsdp":
            strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
        fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
        self.fabric = fabric

        fabric.launch()

        fabric.print(f"Loading model {str(checkpoint_dir)!r}", file=sys.stderr)
        # check_valid_checkpoint_dir(checkpoint_dir)

        with open(checkpoint_dir / "lit_config.json") as fp:
            config = Config(**json.load(fp))

        if quantize is not None and ((isinstance(devices, list) and len(devices) > 1) or (isinstance(devices, int) and devices > 1)):
            raise NotImplementedError
        if quantize == "gptq.int4":
            model_file = "lit_model_gptq.4bit.pth"
            if not (checkpoint_dir /  model_file).is_file():
                raise ValueError("Please run `python quantize/gptq.py` first")
        else:
            model_file = "lit_model.pth"
        checkpoint_path = checkpoint_dir /  model_file

        fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
        t0 = time.time()

        with fabric.init_module(empty_init=True), quantization(quantize):
            model = GPT(config)
        fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)



        t0 = time.time()
        with lazy_load(checkpoint_path) as checkpoint:
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=quantize is None)
        fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

        model.eval()
        self.model = fabric.setup_module(model)
        self.tokenizer = Tokenizer(checkpoint_dir)
        L.seed_everything(seed)
        # if test:
        #     self.test(model=self)
        self.device = c.copy(fabric.device)

    @classmethod
    def install(cls, install_torch :bool= True , install_flash:bool = False):
        c.cmd('pip install pytorch-lightning', verbose=True)
        c.cmd("pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'", verbose=True)
        c.cmd("pip install 'flash-attn>=2.0.0.post1' --no-build-isolation")
        c.cmd("pip install -U 'bitsandbytes>=0.40.0'")
        return c.cmd('pip install -e ./', cwd=cls.dirpath(), verbose=True)
    
    def download(self, model='vicuna.13b'):
        model = c.model_shortcuts().get(model, model)
        cmd = f'python3 scripts/download.py --repo_id {model}'
        c.cmd(cmd, verbose=True, cwd=self.dirpath())

    @property
    def checkpoint_paths(self):
        return list(self.model2checkpoint.values())
    
    checkpoints = checkpoint_paths

    @property
    def checkpoint_models(self):
        return list(self.model2checkpoint.keys())
    
    @property
    def model2checkpoint(self):
        checkpoint_dirpath = self.dirpath() + '/checkpoints/'

        import os
        path2model_id = lambda x: os.path.dirname(x).replace(checkpoint_dirpath, '')
        return {path2model_id(p): os.path.dirname(p) for p in c.walk(checkpoint_dirpath) if p.endswith('lit_config.json')}
    

    
    @torch.no_grad()
    def generate(self,
        prompt: str,
        max_new_tokens: int = 10,
        max_seq_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        return_dict : bool = True,
        **kwargs
    ) -> torch.Tensor:
        
        """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        The implementation of this function is modified from A. Karpathy's nanoGPT.

        Args:
            idx: Tensor of shape (T) with indices of the prompt sequence.
            max_returned_tokens: The maximum number of tokens to return (given plus generated).
            max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
            temperature: Scales the predicted logits by 1 / temperature.
            top_k: If specified, only sample among the tokens with the k highest probabilities.
            eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        """
        model = self.model
        t_start = c.time()
        encoded = self.tokenizer.encode(prompt)

        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        assert max_returned_tokens <= model.config.block_size, (
            max_returned_tokens,
            model.config.block_size,
        )  # maximum rope cache length

        if len(encoded) > max_seq_length:
            encoded = encoded[-max_seq_length:]
        idx = encoded

        T = idx.size(0)
        assert max_returned_tokens > T
        device, dtype = idx.device, idx.dtype
        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
        empty[:T] = idx
        idx = empty
        input_pos = torch.arange(0, T, device=device)


        idx = self.fabric.to_device(idx)
        input_pos = self.fabric.to_device(input_pos)


        if idx.device.type == "xla":
            import torch_xla.core.xla_model as xm

            xm.mark_step()
        # generate up to a fixed number of tokens
        for _ in range(max_returned_tokens - T):
            x = idx.index_select(0, input_pos).view(1, -1)

            c.print(x, input_pos, idx, prompt_length, device)

            # forward
            logits =self.model(x, max_seq_length, input_pos)
            logits = logits[0, -1] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

            # advance
            input_pos = input_pos[-1:] + 1

            if idx.device.type == "xla":
                xm.mark_step()

            # concatenate the new generation
            idx = idx.index_copy(0, input_pos, idx_next)

            # if <eos> token is triggered, return the output (stop generation)
            if idx_next == eos_id:
                idx =  idx[:input_pos]  # include the EOS token
                break

        output_text =  self.tokenizer.decode(idx)
        if return_dict:
            output = {
                'text': output_text,
                'input_tokens': prompt_length,
                'output_tokens': idx.size(0) - prompt_length,
                'time': c.round(c.time() - t_start, 4),
            }
            output['tokens_per_second'] = c.round(output['output_tokens'] / output['time'], 4)

            return output
  
        return text
    
    chat = talk = generate


    def resolve_model(self, model) -> str:
        return c.resolve_model_shortcut(model)
    
    def get_model_checkpoint(self, model):
        model = self.resolve_model(model)
        checkpoint_dir = self.model2checkpoint[model]
        return checkpoint_dir
    
    def checkpoint_exists(self, model):
        model = self.resolve_model(model)
        return model in self.model2checkpoint
    

    @classmethod
    def test(cls, module=None,  
                prompt = 'What is the difference between an einsum and a matrix multiplicaation?',
                num_samples = 2,
                max_new_tokens=10,
                **kwargs):
        if module is None:
            module = cls(test=False)
        for i in range(num_samples):
            output = module.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            c.print(output)
            excepted_keys = ['text', 'input_tokens', 'output_tokens', 'time', 'tokens_per_second']
            for k in excepted_keys:
                assert k in output, f'{k} not in output'
        return output


    @classmethod
    def test_remote(cls, module=None,  
                prompt = 'What is the difference between an einsum and a matrix multiplicaation?',
                num_samples = 2,
                max_new_tokens=10,
                **kwargs):
        for i in range(num_samples):
            output = module.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
            c.print(output)
            excepted_keys = ['text', 'input_tokens', 'output_tokens', 'time', 'tokens_per_second']
            for k in excepted_keys:
                assert k in output, f'{k} not in output'
        return output

    @classmethod
    def serve(cls, model='vicuna.7b', 
                remote=False,
                refresh=False,
                tag=None,  
                devices=None, 
                  **kwargs):

        if '::' in model:
            # extract model name from path
            name = c.copy(model)
            model = '::'.join(model.split('::')[:-1])
        else:
            name = f'model.{model}'

        if tag is not None:
            name = f'{name}::{tag}'
        kwargs['model'] = model

        # resolve the device
        if devices == None:
            devices = c.model_max_gpus(model)
        kwargs['devices'] = devices
        c.serve(module=cls.module_path(), name=name, kwargs=kwargs, remote=remote, refresh=refresh)

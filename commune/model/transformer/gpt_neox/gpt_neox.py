import commune
from typing import Optional, Tuple, Dict, List
import torch
from torch import nn
from commune.model.transformer.gpt_neox.gpt_neox_blocks import GPTNeoXLayer
import streamlit as st
class GPTNeox(commune.Module, nn.Module):
    
    def __init__(self,layer=GPTNeoXLayer, **kwargs):
        
        self.layer = layer
        nn.Module.__init__(self)
        config = self.set_config(kwargs=kwargs)
        self.set_model(self.config)
    def set_model(self, config):
        if config.init_empty_weights:
            with self.init_empty_weights():
                config.init_empty_weights = False
                model = self.set_model(config)
            return model
        self.embed_in = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        

        
        layers = []
        config['block_info'] = block_info = {}
        
        
        # IN BLOCK
        
        block_info['in'] = {
            'params':  self.get_num_params(self),
            'size': self.get_model_size(self)
        }
        
        
        # N BLOCKS
        block_info['layers'] = []  
        max_gpu_memory = {}
        for i in range(config.num_hidden_layers):
            
            layer = self.layer(config)
            layers.append(layer)
            
            block_info['layers'] += [{
                'params':  self.get_num_params(layer),
                'size': self.get_model_size(layer),
                'layer': type(layer).__name__,
            }]
        

        free_gpu_memory = self.free_gpu_memory(max_gpu_ratio=self.config.max_gpu_ratio)
        
        # OUT BLOCK
    
        self.final_layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.embed_in = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.gradient_checkpointing = False
        
        block_info['out'] = {
            'params':  self.get_num_params(self)-  block_info['in']['params'],
            'size': self.get_model_size(self) -  block_info['in']['size']
        }
        
        next_gpu = self.most_free_gpu(free_gpu_memory)
        next_gpu_memory = free_gpu_memory[next_gpu]
        non_middle_memory = sum([block_info[block]['size'] for block in ['in', 'out']])
        assert next_gpu_memory > non_middle_memory
        free_gpu_memory[next_gpu] -= non_middle_memory
        for block in ['in', 'out']:
            block_info[block]['gpu'] = next_gpu
        # self.to(next_gpu)
        
        for i, layer in enumerate(block_info['layers']):
            if free_gpu_memory[next_gpu] < layer['size']:
                next_gpu = self.most_free_gpu(free_gpu_memory)
            free_gpu_memory[next_gpu] -= layer['size']
            layer['gpu'] = next_gpu
        
        
        config['block_info'] = block_info
        st.write(config.block_info)

        # self.to(next_gpu)
        self.device = next_gpu
        
        
        self.layers = nn.ModuleList(layers)


        if config.init_weights:
            self.init_weights()
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.load_weights(config.load_weights)
        self.config = config

    

    base = GPTNeoXLayer
    def resolve_block(self, block:str):
        if isinstance(block, str):
            block = commune.connect(block)
        elif isinstance(block, self.base ):
            block = block
        else:
            raise ValueError(f"block must be a string or a {self.base} object")
        return self.layers[index]
    
    def replace_block(self, index, layer):
        self.layers[index] = layer

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention layers. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for layer_past
                        return module(*inputs, use_cache, None, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return dict(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        
    @classmethod
    def test(cls):
        with cls.init_empty_weights():
            self = cls()
        cls.print(self)
        
    def init_weights(self):
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def load_weights(self, model=None):
        hf = self.module('huggingface')
        model = self.config.model if model is None else model
        new_state_dict =  hf.load_model_weights(model)
        state_dict = self.state_dict()
        for k in new_state_dict.keys():
            if k in state_dict.keys():
                state_dict[k] = new_state_dict[k]
        self.load_state_dict(state_dict)
        


if __name__ == "__main__":
    GPTNeox.run()

# print(models)
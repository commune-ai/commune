import commune
from typing import Optional, Tuple, Dict, List
import torch
from torch import nn
from commune.model.transformer.gpt_neox.gpt_neox_blocks import GPTNeoXLayer
import streamlit as st
class GPTNeox(commune.Module, nn.Module):
    
    def __init__(self, **kwargs):
        
        nn.Module.__init__(self)
        config = self.set_config(kwargs=kwargs)
        self.set_model(self.config)
        
        
        
        
    def init_nn(self):
        from torch import nn
        nn.Module.__init__(self)
    LayerModule = GPTNeoXLayer
    
    
    @classmethod
    def quantize(cls,
                 model:str,
                 dynamic_q_layer : set = {torch.nn.Linear}, 
                 dtype=torch.qint8, **kwargs):
        self = torch.ao.quantization.quantize_dynamic( model,  # the original model
        dynamic_q_layer,  # a set of layers to dynamically quantize
        dtype=torch.qint8)
        return self
    

    
    def set_model(self, config):
        if config.init_empty_weights:
            with self.init_empty_weights():
                config.init_empty_weights = False
                model = self.set_model(config)
            return model
        self.embed_in = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        config['blocks'] = blocks = {}
        # IN BLOCK
        
        blocks['in'] = {
            'params':  self.get_num_params(self),
            'size': self.get_model_size(self)
        }
        
        # N BLOCKS
        blocks['layers'] = []  
        layers = []
        for i in range(config.num_hidden_layers):
            layer = self.LayerModule(config)
            layers.append(layer)
            blocks['layers'] += [{
                'params':  self.get_num_params(layer),
                'size': self.get_model_size(layer),
                'layer': type(layer).__name__,
            }]
        

        
        self.final_layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

        blocks['out'] = {
            'params':  self.get_num_params(self)-  blocks['in']['params'],
            'size': self.get_model_size(self) -  blocks['in']['size']
        }
        
        
        free_gpu_memory = self.free_gpu_memory(max_gpu_ratio=self.config.max_gpu_ratio)

        
        # self.print('broo')
        next_gpu = self.most_free_gpu(free_gpu_memory)
        next_gpu_memory = free_gpu_memory[next_gpu]
        non_middle_memory = sum([blocks[block]['size'] for block in ['in', 'out']])
        assert next_gpu_memory > non_middle_memory
        free_gpu_memory[next_gpu] -= non_middle_memory
        
        for b in ['in', 'out']:
            blocks[b]['device'] = f'cuda:{next_gpu}'
            
            
        self.device = blocks['in']['device']
        self.to(self.device)
        
        self.layers = nn.ModuleList(layers)
        for i, layer in enumerate(blocks['layers']):
            while  layer['size'] > free_gpu_memory[next_gpu]:
                print(free_gpu_memory[next_gpu], layer['size'])
                next_gpu = self.most_free_gpu(free_gpu_memory)
            free_gpu_memory[next_gpu] -= layer['size']
            blocks['layers'][i]['device'] = f'cuda:{next_gpu}'
            
            self.print(f"Layer {i} on {blocks['layers'][i]['device']}, {free_gpu_memory[next_gpu]} left")
            self.layers[i].to(blocks['layers'][i]['device'])
        
        config['blocks'] = blocks

        if config.init_weights:
            self.init_weights()
            
        self.load_weights(config.load_weights)
        self.config = config


        if config.quantize:
            self.quantize(model=self)
    


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
        no_grad : bool = False,
        autocast: bool = False,
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
        

        kwargs = self.locals2kwargs(locals())
        if kwargs.get('no_grad', False):
            kwargs['no_grad'] = False
            with torch.no_grad():
                return self.forward(**kwargs)
            
        if kwargs.get('autocast', False):
            kwargs['autocast'] = False
            with torch.cuda.amp.autocast():
                return self.forward(**kwargs)
        
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

        input_ids = input_ids.to(self.device)
        
        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)


        position_ids = None
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()


        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.to(self.device)
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

            
            layer_kwargs = dict(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                position_ids=position_ids,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        
            layer_device = self.config.blocks['layers'][i]['device']

            for k, v in layer_kwargs.items():
                if isinstance(v, torch.Tensor):
                    layer_kwargs[k] = v.to(layer_device)
            
            outputs = layer(**layer_kwargs)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)


        out_block_device = self.config.blocks['out']['device']
        
        hidden_states = self.final_layer_norm(hidden_states.to(out_block_device))
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
        

    def get_head_mask(
        self, head_mask: Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> torch.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    @staticmethod
    def dict_schema(x) -> dict:
        dict_schema = {}
        for k,v in x.items():
            v_schema = {}
            v_schema['type'] = type(v)
            if v_schema['type'] == torch.Tensor:
                v_schema['shape'] = v.shape
                v_schema['dtype'] = v.dtype
                v_schema['device'] = v.device
                
            if v_schema['type'] == dict:
                v_schema = dict_schema(v)
            dict_schema[k] = v_schema
            
    
        return dict_schema
    
    @classmethod
    def test(cls):
        c = commune
        c.new_event_loop()
        model = GPTNeox()

        sample = c.call('dataset.bittensor', fn='sample')
        
        with torch.no_grad():
            output =  model.forward(**sample)
        cls.print(cls.dict_schema(output))     

if __name__ == "__main__":

    GPTNeox.test()

# print(models)
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from torch import nn
from itertools import zip_longest
from .resnet import TemporalConvLayer, Downsample2D, ResnetBlock2D, Upsample2D
from .transformers import Transformer2DModel, TransformerTemporalModel, TransformerTemporalConditioningModel

def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    only_cross_attention=False,
    upcast_attention=False
):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    only_cross_attention=False,
    upcast_attention=False
):
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention
        )
    raise ValueError(f"{up_block_type} does not exist.")

class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        upcast_attention=False,
    ):
        super().__init__()

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=dropout
            )
        ]
        attentions = []
        temp_attentions = []
        temp_conditioning_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    in_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    in_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
            temp_conditioning_attentions.append(
                TransformerTemporalConditioningModel(
                    in_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=True
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        self.temp_conditioning_attentions = nn.ModuleList(temp_conditioning_attentions)

    def forward(
        self,
        hidden_states,
        conditioning_hidden_states,
        h_emb=None,
        c_emb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None
    ):
        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            
            hidden_states, conditioning_hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.resnets[0]), hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
            hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.temp_convs[0]), hidden_states, num_frames) if num_frames > 1 else hidden_states
        else:
            hidden_states, conditioning_hidden_states = self.resnets[0](hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
            hidden_states = self.temp_convs[0](hidden_states, num_frames) if num_frames > 1 else hidden_states
            
        for attn, temp_attn, temp_cond_attn, resnet, temp_conv in zip_longest(
            self.attentions, self.temp_attentions, self.temp_conditioning_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            if self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states,)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_attn, return_dict=False), hidden_states, num_frames)[0] if num_frames > 1 else hidden_states
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_cond_attn, return_dict=False), hidden_states, conditioning_hidden_states, num_frames)[0] if num_frames > 1 else hidden_states
                hidden_states, conditioning_hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_conv), hidden_states, num_frames) if num_frames > 1 else hidden_states
            else:
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states,cross_attention_kwargs=cross_attention_kwargs,).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample if num_frames > 1 else hidden_states
                hidden_states = temp_cond_attn(hidden_states, conditioning_hidden_states, num_frames=num_frames).sample if num_frames > 1 else hidden_states
                hidden_states, conditioning_hidden_states = resnet(hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames=num_frames)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames) if num_frames > 1 else hidden_states    

        return hidden_states, conditioning_hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()

        resnets = []
        attentions = []
        temp_attentions = []
        temp_conditioning_attentions = []
        temp_convs = []

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=dropout
                )
            )
            attentions.append(
                Transformer2DModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
            temp_conditioning_attentions.append(
                TransformerTemporalConditioningModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=True
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        self.temp_conditioning_attentions = nn.ModuleList(temp_conditioning_attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states,
        conditioning_hidden_states,
        h_emb=None,
        c_emb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        output_states = ()
        conditioning_output_states = ()

        for resnet, temp_conv, attn, temp_attn, temp_cond_attn in zip_longest(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions, self.temp_conditioning_attentions
        ):
        
            if self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                
                hidden_states, conditioning_hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_conv), hidden_states, num_frames) if num_frames > 1 else hidden_states
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states,)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_attn, return_dict=False), hidden_states, num_frames)[0] if num_frames > 1 else hidden_states
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_cond_attn, return_dict=False), hidden_states, conditioning_hidden_states, num_frames)[0] if num_frames > 1 else hidden_states
            else:
                hidden_states, conditioning_hidden_states = resnet(hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames=num_frames)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames) if num_frames > 1 else hidden_states
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs,).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample if num_frames > 1 else hidden_states
                hidden_states = temp_cond_attn(hidden_states, conditioning_hidden_states, num_frames=num_frames).sample if num_frames > 1 else hidden_states

            output_states += (hidden_states,)
            conditioning_output_states += (conditioning_hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states, conditioning_hidden_states = downsampler(hidden_states, conditioning_hidden_states)

            output_states += (hidden_states,)
            conditioning_output_states += (conditioning_hidden_states,)

        return hidden_states, output_states, conditioning_hidden_states, conditioning_output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()

        resnets = []
        temp_convs = []

        self.gradient_checkpointing = False
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, conditioning_hidden_states, h_emb=None, c_emb=None, num_frames=1):
        output_states = ()
        conditioning_output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            if self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states, conditioning_hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_conv), hidden_states, num_frames) if num_frames > 1 else hidden_states
            else:
                hidden_states, conditioning_hidden_states = resnet(hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames=num_frames)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames) if num_frames > 1 else hidden_states                  

            output_states += (hidden_states,)
            conditioning_output_states += (conditioning_hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states, conditioning_hidden_states = downsampler(hidden_states, conditioning_hidden_states)

            output_states += (hidden_states,)
            conditioning_output_states += (conditioning_hidden_states,)

        return hidden_states, output_states, conditioning_hidden_states, conditioning_output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()

        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []
        temp_conditioning_attentions = []

        self.gradient_checkpointing = False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=dropout
                )
            )
            attentions.append(
                Transformer2DModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                )
            )
            temp_conditioning_attentions.append(
                TransformerTemporalConditioningModel(
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=True
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        self.temp_conditioning_attentions = nn.ModuleList(temp_conditioning_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        conditioning_hidden_states,
        res_conditioning_hidden_states_tuple,
        h_emb=None,
        c_emb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        num_frames=1,
        cross_attention_kwargs=None,
    ):
        for resnet, temp_conv, attn, temp_attn, temp_cond_attn in zip_longest(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions, self.temp_conditioning_attentions
        ):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            res_conditioning_hidden_states = res_conditioning_hidden_states_tuple[-1]
            res_conditioning_hidden_states_tuple = res_conditioning_hidden_states_tuple[:-1]
            conditioning_hidden_states = torch.cat([conditioning_hidden_states, res_conditioning_hidden_states], dim=1)

            if self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                
                hidden_states, conditioning_hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_conv), hidden_states, num_frames) if num_frames > 1 else hidden_states
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states,)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_attn, return_dict=False), hidden_states, num_frames)[0] if num_frames > 1 else hidden_states
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_cond_attn, return_dict=False), hidden_states, conditioning_hidden_states, num_frames)[0] if num_frames > 1 else hidden_states
            else:
                hidden_states, conditioning_hidden_states = resnet(hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames=num_frames)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames) if num_frames > 1 else hidden_states
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs,).sample
                hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample if num_frames > 1 else hidden_states
                hidden_states = temp_cond_attn(hidden_states, conditioning_hidden_states, num_frames=num_frames).sample if num_frames > 1 else hidden_states

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states, conditioning_hidden_states = upsampler(hidden_states, conditioning_hidden_states, upsample_size)

        return hidden_states, conditioning_hidden_states

class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()

        resnets = []
        temp_convs = []
        self.gradient_checkpointing = False

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=dropout
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
            self, 
            hidden_states, 
            res_hidden_states_tuple, 
            conditioning_hidden_states, 
            res_conditioning_hidden_states_tuple, 
            h_emb=None,
            c_emb=None,
            upsample_size=None, 
            num_frames=1
        ):
        
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            res_conditioning_hidden_states = res_conditioning_hidden_states_tuple[-1]
            res_conditioning_hidden_states_tuple = res_conditioning_hidden_states_tuple[:-1]
            conditioning_hidden_states = torch.cat([conditioning_hidden_states, res_conditioning_hidden_states], dim=1)

            if self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states, conditioning_hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(temp_conv), hidden_states, num_frames) if num_frames > 1 else hidden_states
            else:
                hidden_states, conditioning_hidden_states = resnet(hidden_states, conditioning_hidden_states, h_emb, c_emb, num_frames=num_frames)
                hidden_states = temp_conv(hidden_states, num_frames=num_frames) if num_frames > 1 else hidden_states

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states, conditioning_hidden_states = upsampler(hidden_states, conditioning_hidden_states, upsample_size)

        return hidden_states, conditioning_hidden_states

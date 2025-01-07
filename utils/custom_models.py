import pdb
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Dict, Optional
from copy import deepcopy


def convert_openclip_to_dino(openclip_model):
    # This will return the intersection between dino and openclip weights
    if isinstance(openclip_model, dict):
        sd = openclip_model
    else:
        sd = openclip_model.state_dict()
    converted_sd = {}
    for key, weight in sd.items():
        add_key = False
        if key.startswith('transformer.resblocks'):
            key = key.replace('transformer.resblocks.', 'encoder.layer.')
        
        # Take care of attention
        if 'attn.attention' in key:
            key = key.replace('attn.attention', 'attention.attention')
            add_key = True
        if 'attn.merge_heads.out_proj' in key:
            key = key.replace('attn.merge_heads.out_proj', 'attention.output.dense')
            add_key = True
        # Take care of layernorms
        if 'ln_1' in key:
            key = key.replace('ln_1', 'layernorm_before')
            add_key = True
        if 'ln_2' in key:
            key = key.replace('ln_2', 'layernorm_after')
            add_key = True
        # Take care of MLPs
        if 'mlp.c_fc' in key:
            key = key.replace('mlp.c_fc', 'intermediate.dense')
            add_key = True
        if 'mlp.c_proj' in key:
            key = key.replace('mlp.c_proj', 'output.dense')
            add_key = True
        
        # Take care of auxiliary layers
        if 'class_embedding' == key:
            key = 'embeddings.cls_token'
            weight = weight.reshape(1, 1, -1)
            add_key = True
        if key.startswith('ln_post'):
            key = key.replace('ln_post', 'layernorm')
            add_key = True
        if key == 'conv1.weight':
            key = 'embeddings.patch_embeddings.projection.weight'
            add_key = True
        if key.startswith('proj'):
            key = 'pooler.dense.weight'
            weight = weight.T
            add_key = True
        if key.startswith('ln_pre'):
            key = key.replace('ln_pre', 'embeddings.ln_pre')
            add_key = True
        if key == 'positional_embedding':
            key = 'embeddings.position_embeddings'
            weight = weight[None]
            add_key = True
        
        if add_key:
            converted_sd[key] = weight
        else:
            print('skipped, ', key)
    return converted_sd


def lnpre_forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.ln_pre(embeddings)

        return embeddings
    

def ready_dino_as_clip(dino_model):
    dino_model.pooler.dense.weight.data = dino_model.pooler.dense.weight.data[:512]         # Make output projection size 512
    dino_model.pooler.dense.bias.data = dino_model.pooler.dense.bias.data[:512]             # Make output projection size 512
    dino_model.pooler.activation = nn.Identity()                                                  # Remove Tanh activation
    dino_model.embeddings.ln_pre = nn.LayerNorm(768)                                              # Add pre layer norm
    dino_model.embeddings.forward = types.MethodType(lnpre_forward, dino_model.embeddings)  # integrate pre layer norm into forward
    return dino_model


def subselect_from_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    if any(['visual.model' in key for key in state_dict_keys]):
        new_state_dict = {}
        for key in state_dict_keys:
            if 'visual.model' in key:
                new_state_dict[key.replace('visual.model.', '')] = state_dict[key]
        return new_state_dict
    else:
        return state_dict

def correct_state_dict(sd):
    state_dict = subselect_from_state_dict(sd)
    new_state_dict = {}
    # pdb.set_trace()
    for k, v in state_dict.items():
        new_k = k.replace('model.', '')
        new_k = new_k.replace('compute_mha', 'attention')
        new_k = new_k.replace('attn.out_proj', 'attn.merge_heads.out_proj')
        # new_k = new_k.replace('compute_mha.', '')
        # new_k = new_k.replace('attn.compute_mha', 'attn')
        # new_k = new_k.replace('attn.attention', 'attn')
        # new_k = new_k.replace('attn.merge_heads', 'attn')
        new_state_dict[new_k] = v
    return new_state_dict


def convert_clip_to_dino(dino_model, clip_path):
    dino_model = ready_dino_as_clip(dino_model)
    openclip_sd = torch.load(clip_path, map_location=dino_model.device)
    # pdb.set_trace()
    if any(['visual.model' in key for key in openclip_sd.keys()]):
        openclip_sd = correct_state_dict(openclip_sd)
    # Check if the model is already a CLIPlike DiNO model
    if len(set(list(openclip_sd.keys())) - set(list(dino_model.state_dict().keys()))) == 0:
        return load_clip_into_dino(dino_model, clip_path)
    # Convert the openclip model to dino model
    clip_converted_sd = convert_openclip_to_dino(openclip_sd)
    joint_sd = deepcopy(clip_converted_sd)
    dino_sd = dino_model.state_dict()
    joint_sd.update(
        {
            'embeddings.patch_embeddings.projection.bias': torch.zeros_like(dino_sd['embeddings.patch_embeddings.projection.bias']),
            'pooler.dense.bias': torch.zeros_like(joint_sd['pooler.dense.weight'][:, 0]),
        }
    )
    
    out = dino_model.load_state_dict(joint_sd)
    print(out)
    return dino_model


def load_clip_into_dino(dino_model, clip_path):
    print("Loading CLIP weights into DINO model. From ", clip_path)
    clip_sd = torch.load(clip_path, map_location=dino_model.device)
    pdb.set_trace()
    dino_model = ready_dino_as_clip(dino_model)
    out = dino_model.load_state_dict(clip_sd)
    print(out)
    return dino_model


def check_sd_alignment(sd0, sd1, atol=1e-7):
    aligned = []
    unaligned = []
    for k, v0 in sd0.items():
        v1 = sd1[k]
        if not torch.allclose(v1, v0, atol=atol):
            unaligned.append(k)
        else:
            aligned += [k]
    return {
        'aligned': aligned,
        'unaligned': unaligned
    }
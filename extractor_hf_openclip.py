import argparse
import torch
import torchvision.transforms
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import torch.nn.functional as F
import math
import timm
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
import os
import open_clip
import pdb
from typing import Optional
from copy import deepcopy
from custom_attention import AttentionBlock


class Attention(nn.Module):
    # copied directly from openclip backend. Because nn.MultiheadAttention is really annoying
    def __init__(
            self,
            module: torch.nn.Module,
    ):
        super().__init__()
        if hasattr(module, 'in_proj_bias'):
            self.bias = True

        out_dim, in_dim = module.in_proj_weight.shape
        hidden_dim = out_dim // 3
        self.query = nn.Linear(in_dim, hidden_dim, bias=self.bias)
        self.key = nn.Linear(in_dim, hidden_dim, bias=self.bias)
        self.value = nn.Linear(in_dim, hidden_dim, bias=self.bias)
        
        # replace weights
        self.query.weight.data = module.in_proj_weight[:hidden_dim, :]
        self.key.weight.data = module.in_proj_weight[hidden_dim:2*hidden_dim, :]
        self.value.weight.data = module.in_proj_weight[2*hidden_dim:, :]
        if self.bias:
            self.query.bias.data = module.in_proj_bias[:hidden_dim]
            self.key.bias.data = module.in_proj_bias[hidden_dim:2*hidden_dim]
            self.value.bias.data = module.in_proj_bias[2*hidden_dim:]
            
        self.out_proj = module.out_proj
        self.out_drop = nn.Dropout(module.dropout)
        self.batch_first = module.batch_first
        self.num_heads = module.num_heads
        
        self.debugging_module = module

    def transpose_tokens(self, x):
        x = x.transpose(0, 1)
    
    def forward(
        self, q_x, k_x: Optional[torch.Tensor]=None, v_x: Optional[torch.Tensor]=None, 
        attn_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        # x = deepcopy(q_x) # [B,T,C]
        if k_x is None:
            k_x = deepcopy(q_x)
            v_x = deepcopy(q_x)
            
        if not self.batch_first:
            for elem in [q_x, k_x, v_x]:
                self.transpose_tokens(elem)
        
        q = self.query(q_x)
        k = self.key(k_x)
        v = self.value(v_x)
        
        q = q.transpose(0, 1) # [T,B,C]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        T, B, C = q.shape
        q = q.reshape(T, B * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(T, B * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(T, B * self.num_heads, -1).transpose(0, 1) # [BxnH,T,C/nH]

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            pdb.set_trace()
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        attn = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        assert attn.shape[-1] == T
        # attn = self.attn_drop(attn)
        x = torch.bmm(attn, v)
        x = x.transpose(1, 0).reshape(T, B, self.num_heads, -1).flatten(2)
        # x = x.reshape(T, B, 

        if not self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        pdb.set_trace()
        debugging_x = self.debugging_module(q_x, k_x, v_x, attn_mask=attn_mask, **kwargs)
        if not torch.allclose(debugging_x[0], x):
            pdb.set_trace()
        # pdb.set_trace()
        return (x, None)
    

def recursively_setattr(model, key, new_module):
        stages = key.split('.')
        x = getattr(model, stages[0])
        for stage in stages[1:-1]:
            if stage in [str(i) for i in range(20)]:
                x = x[int(stage)]
                continue
            x = getattr(x, stage)
        # pdb.set_trace()
        setattr(x, stages[-1], new_module)


def replace_attention_layers(model, attention_layers):
    for key, module in attention_layers.items():
        if isinstance(module, torch.nn.Identity):
            pdb.set_trace()
        new_module = Attention(module)
        # new_module = AttentionBlock(module)
        recursively_setattr(model, key, new_module)
    return model


class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.
    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self, model_type: tuple =('ViT-B-16', 'laion400m_e31'), 
        stride: int = 4, model: nn.Module = None, 
        device: str = 'cuda',
        model_load_path: Optional[str] = None,
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = self.create_model(model_type, model_load_path)
            # pdb.set_trace()
            # TODO: DELETE THESE LINES
            # temp_sd = torch.load(
            #     '/mmfs1/gscratch/krishna/gstoica3/research/sd-dino/checkpoints/clip_key_merged_with_dino.pth'
            # )
            # self.model.transformer.resblocks[9].attn.key.weight.data = (
            #     temp_sd['transformer.resblocks.9.attn.key.weight']
            # )
            # self.model.transformer.resblocks[9].attn.key.bias.data = (
            #     temp_sd['transformer.resblocks.9.attn.key.bias']
            # )
            ##################################################################
            # self.model = self.model
        
        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_size
        if type(self.p)==tuple:
            self.p = self.p[0]
        self.stride = self.model.conv1.stride

        
        self.processor = open_clip.create_model_and_transforms(model_type[0], pretrained=model_type[1])[-1]
        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None
        
    def subselect_from_state_dict(self, state_dict):
        state_dict_keys = list(state_dict.keys())
        if any(['visual.model' in key for key in state_dict_keys]):
            new_state_dict = {}
            for key in state_dict_keys:
                if 'visual.model' in key:
                    new_state_dict[key.replace('visual.model.', '')] = state_dict[key]
            return new_state_dict
        else:
            return state_dict
    
    def correct_state_dict(self, load_path):
        state_dict = self.subselect_from_state_dict(torch.load(load_path, map_location='cuda:0'))
        new_state_dict = {}
        # pdb.set_trace()
        for k, v in state_dict.items():
            new_k = k.replace('model.', '')
            # new_k = new_k.replace('compute_mha', 'attention')
            # new_k = new_k.replace('attn.out_proj', 'attn.merge_heads.out_proj')
            # new_k = new_k.replace('compute_mha.', '')
            new_k = new_k.replace('attn.attention', 'attn')
            new_k = new_k.replace('attn.merge_heads', 'attn')
            new_state_dict[new_k] = v
        return new_state_dict

    
    def create_model(self, model_type: str, load_path: Optional[str]=None) -> nn.Module:
        model =  open_clip.create_model_and_transforms(model_type[0], pretrained=model_type[1])[0].visual
        attention_layers = {key: module for key, module in model.named_modules() if key.endswith('.attn')}
        model = replace_attention_layers(model, attention_layers)
        if load_path is not None:
            out = model.load_state_dict(self.correct_state_dict(load_path), strict=False)
            print(out)
            print("Loaded model from ", load_path)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            positional_embedding = self.positional_embedding.T
            npatch = x.shape[1] - 1
            N = positional_embedding.shape[1] - 1
            if npatch == N and w == h:
                return positional_embedding
            class_pos_embed = positional_embedding[:, 0].unsqueeze(0)
            patch_pos_embed = positional_embedding[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            # pdb.set_trace()
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding
    
    @staticmethod
    def interpolatable_forward(self, x):
        batch_size, num_channels, height, width = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        x = x + self.interpolate_pos_encoding(x, height, width)
        x = self.ln_pre(x)
        # pdb.set_trace()
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        # pdb.set_trace()
        patch_size = model.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.conv1.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        model.forward = types.MethodType(ViTExtractor.interpolatable_forward, model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None, patch_size: int = 14) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        def divisible_by_num(num, dim):
            return num * (dim // num)
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)

            width, height = pil_image.size
            new_width = divisible_by_num(patch_size, width)
            new_height = divisible_by_num(patch_size, height)
            pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)
            
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def preprocess_pil(self, pil_image):
        # pdb.set_trace()
        return self.processor(pil_image)[None, ...]

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output.permute(1, 0, 2))
            return _hook

        # if facet == 'key':
        #      def _inner_hook(module, input, output):
        #         input = input[0]#.permute(1, 0, 2)
        #         N, B, C = input.shape
        #         qkv_weight = module.in_proj_weight
        #         qkv_bias = module.in_proj_bias
        #         outs = torch.nn.functional.linear(input, qkv_weight[:C, :], qkv_bias[:C]).reshape(N, B, module.num_heads, -1).permute(1, 2, 0, 3)
        #         self._feats.append(outs)
        
        # elif facet == 'query':
        #      def _inner_hook(module, input, output):
        #         input = input[0]#.permute(1, 0, 2)
        #         N, B, C = input.shape
        #         qkv_weight = module.in_proj_weight
        #         qkv_bias = module.in_proj_bias
        #         outs = torch.nn.functional.linear(input, qkv_weight[C:2*C], qkv_bias[C:2*C]).reshape(N, B, module.num_heads, -1).permute(1, 2, 0, 3)
        #         self._feats.append(outs)
            
        # elif facet == 'value':
        #      def _inner_hook(module, input, output):
        #         input = input[0]#.permute(1, 0, 2)
        #         # print("The shape of OpenCLIP the qkv hook inputs is: ", input.shape)
        #         N, B, C = input.shape
        #         qkv_weight = module.in_proj_weight
        #         qkv_bias = module.in_proj_bias
        #         outs = torch.nn.functional.linear(input, qkv_weight[2*C:], qkv_bias[2*C:]).reshape(N, B, module.num_heads, -1).permute(1, 2, 0, 3)
        #         # print("The shape of OpenCLIP the qkv hook outs is: ", outs.shape)
        #         # print("outs is: ", outs)
        #         self._feats.append(outs)
            
            
            
        
        if facet in ['query', 'key', 'value']:
            def _inner_hook(module, input, output):
                # input = input[0].permute(1, 0, 2)
                # qkv_weight = module.in_proj_weight
                # qkv_bias = module.in_proj_bias
                # outs = torch.nn.functional.linear(input, qkv_weight[:C, :], qkv_bias[:C]).reshape(N, B, module.num_heads, -1).permute(1, 2, 0, 3)
                print(output.shape)
                B, N, C = output.shape
                # [H, T, B, D]. Should be: # [B, H, T, C]
                # 1, 12, 2810, 64
                # outs = output.reshape(B, N, 12, -1).permute(1, 2, 0, 3)
                outs = output.reshape(B, N, 12, -1).transpose(1, 2)
                print("The shape of OpenCLIP the qkv hook outputs is: ", outs.shape)
                self._feats.append(outs)
        # elif facet == 'qkv':
        #     def _inner_hook(module, input, output):
        #         input = input[0]
        #         N, B, C = input.shape
        #         qkv_weight = module.in_proj_weight
        #         qkv_bias = module.in_proj_bias
        #         # print("The shape of OpenCLIP the qkv hook inputs is: ", input.shape)
        #         outputs = torch.nn.functional.linear(input, qkv_weight, qkv_bias).reshape(N, B, 3, module.num_heads, -1).permute(1, 3, 0, 2, 4).flatten(3, 4)
        #         # print("The shape of OpenCLIP the qkv hook outputs is: ", outputs.shape)
        #         self._feats.append(outputs)
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        # def _inner_hook(module, input, output):
        #     input = input[0]
        #     B, N, C = input.shape
        #     qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        #     self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        Assumption is that we are in the ViTLayer block.
        attn: the actual attention softmax values
        key: the key values
        query: the query values
        value: the value values
        token: the output of the entire block (including the MLP)
        
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        # pdb.set_trace()
        for block_idx, block in enumerate(self.model.transformer.resblocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                    self.hook_module_name = f'transformer.resblocks.{block_idx}'
                elif facet == 'attn':
                    self.hook_handlers.append(block.ln_attn.register_forward_hook(self._get_hook(facet)))
                    self.hook_module_name = f'transformer.resblocks.{block_idx}.attn.out_proj'
                # elif facet in ['key', 'query', 'value']:
                #     # print(block.attn.in_proj_weight.shape)
                #     self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                #     self.hook_module_name = f'transformer.resblocks.{block_idx}.attn.{facet}'
                elif facet == 'key':
                    try:
                        self.hook_handlers.append(block.attn.key.register_forward_hook(self._get_hook(facet)))
                    except:
                        self.hook_handlers.append(block.attn.attention.key.register_forward_hook(self._get_hook(facet)))
                    self.hook_module_name = f'transformer.resblocks.{block_idx}.attn.{facet}'
                    # pdb.set_trace()
                elif facet == 'query':
                    try:
                        self.hook_handlers.append(block.attn.query.register_forward_hook(self._get_hook(facet)))
                    except:
                        self.hook_handlers.append(block.attn.attention.query.register_forward_hook(self._get_hook(facet)))
                    self.hook_module_name = f'transformer.resblocks.{block_idx}.attn.{facet}'
                elif facet == 'value':
                    try:
                        self.hook_handlers.append(block.attn.value.register_forward_hook(self._get_hook(facet)))
                    except:
                        self.hook_handlers.append(block.attn.attention.value.register_forward_hook(self._get_hook(facet)))
                    self.hook_module_name = f'transformer.resblocks.{block_idx}.attn.{facet}'
                # elif facet == 'qkv':
                #     self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                #     self.hook_module_name = f'transformer.resblocks.{block_idx}.attn'
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        # pdb.set_trace()
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(x=batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token', 'qkv'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            pdb.set_trace()
            x = x.transpose(0, 1).unsqueeze(dim=1) #Bx1xtxd
        # else:
            # x = x.)
        if not (
            x.shape[0] == 1 and \
            x.shape[2] == 2810
        ):
            pdb.set_trace()
        # if facet == 'token':
        #     x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0] #Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor extraction.')
    parser.add_argument('--image_path', type=str, required=True, help='path of the extracted image.')
    parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                              small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--patch_size', default=14, type=int, help="patch size of the model.")
    args = parser.parse_args()

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = ViTExtractor(args.model_type, args.stride, device=device)
        image_batch, image_pil = extractor.preprocess(args.image_path, args.load_size, args.patch_size)
        print(f"Image {args.image_path} is preprocessed to tensor of size {image_batch.shape}.")
        descriptors = extractor.extract_descriptors(image_batch.to(device), args.layer, args.facet, args.bin)
        print(f"Descriptors are of size: {descriptors.shape}")
        torch.save(descriptors, args.output_path)
        print(f"Descriptors saved to: {args.output_path}")
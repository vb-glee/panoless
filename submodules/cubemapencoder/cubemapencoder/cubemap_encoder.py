import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#import better_cubemap
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _cubemapencoder as _backend
except ImportError:
    from .backend import _backend

_interp_to_id = {
    'nearest': 0,
    'linear': 1
}

class _cubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    #@custom_fwd
    def forward(ctx, inputs, embeddings, fail_value, interpolation, enable_seamless):

        embeddings = embeddings.contiguous()
        inputs = inputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        outputs = torch.empty([C,B],dtype=embeddings.dtype,device=embeddings.device)

        _backend.cubemap_encode_forward(
            inputs, embeddings, fail_value, outputs, interpolation, enable_seamless,
            B, C, L
        )

        params = torch.tensor([interpolation,enable_seamless], dtype=torch.int64)
        ctx.save_for_backward(inputs, embeddings, params)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, embeddings, params = ctx.saved_variables
        grad_outputs = grad_outputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        grad_embeddings = torch.zeros_like(embeddings)
        grad_inputs = torch.empty_like(inputs)
        grad_fail =  torch.zeros([C], dtype=embeddings.dtype, device=embeddings.device)

        _backend.cubemap_encode_backward(
            grad_outputs, inputs, embeddings,
            grad_embeddings, grad_inputs, grad_fail,
            params[0].item(), params[1].item(),
            B, C, L
        )
        return grad_inputs, grad_embeddings, grad_fail, None, None

cubemap_encode = _cubemap_encode.apply

'''
class _mipcubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,xxx):
        pass # TODO
    @staticmethod
    @custom_bwd
    def backward(ctx,xxx):
        pass

mipcubemap_encode = _mipcubemap_encode.apply
'''

class CubemapEncoder(nn.Module):
    def __init__(self, output_dim = 6, resolution = 256, interpolation='linear'):
        super().__init__()

        self.input_dim = 3
        self.resolution = resolution
        self.output_dim = output_dim

        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        self.seamless = 1

        self.params = nn.ParameterDict({
            'Cubemap_texture': nn.Parameter(torch.rand(6, self.output_dim, resolution, resolution)*10-5), 
            'Cubemap_failv': nn.Parameter(torch.zeros(self.output_dim))
        })
        self.n_elems = 6 * self.output_dim * resolution * resolution + self.output_dim
        
        self.register_buffer("usage_map",
                             torch.zeros((512, 1024), dtype=torch.float32),
                             persistent=True)
    
    def ensure_usage_map(self, H=512, W=1024, device=None):
        if self.usage_map.shape != (H, W):
            new = torch.zeros((H, W), dtype=self.usage_map.dtype, device=self.usage_map.device)
            h, w = min(H, self.usage_map.shape[0]), min(W, self.usage_map.shape[1])
            new[:h, :w] = self.usage_map[:h, :w]
            self.register_buffer("usage_map", new, persistent=True)
        
        if device is not None:
            self.usage_map = self.usage_map.to(device)

    def forward(self, inputs, track_usage=False, usage_weight=None):
        if track_usage is not False:
            w = track_usage if isinstance(track_usage, torch.Tensor) else usage_weight or torch.ones(inputs.shape[0], device=inputs.device)
            
            self.ensure_usage_map(device=inputs.device)
            with torch.no_grad():
                H, W = self.usage_map.shape
                
                dirs = F.normalize(inputs, dim=-1)
                theta = torch.acos(dirs[:, 2].clamp(-1, 1))
                phi = torch.atan2(dirs[:, 1], dirs[:, 0])
                
                u = ((phi + np.pi) / (2 * np.pi)) * W
                v = (theta / np.pi) * (H - 1)
                
                x0, y0 = torch.floor(u).long() % W, torch.floor(v).long().clamp(0, H-1)
                x1, y1 = (x0 + 1) % W, (y0 + 1).clamp(0, H-1)
                wx, wy = u - x0.float(), v - y0.float()

                w00, w10 = (1 - wx) * (1 - wy), wx * (1 - wy)
                w01, w11 = (1 - wx) * wy, wx * wy

                w = w.to(self.usage_map.dtype).view(-1)
                buf = self.usage_map.view(-1)
                buf.index_add_(0, y0 * W + x0, w00 * w)
                buf.index_add_(0, y0 * W + x1, w10 * w)
                buf.index_add_(0, y1 * W + x0, w01 * w)
                buf.index_add_(0, y1 * W + x1, w11 * w)

        outputs = cubemap_encode(inputs, self.params['Cubemap_texture'], self.params['Cubemap_failv'], self.interp_id, self.seamless)
        return outputs.permute(1, 0)

class MipCubemapEncoder(nn.Module):
    def __init__(self, num_levels=4, level_dim=6, per_level_scale=4, base_resolution=4,interpolation='linear',concat=True):
        super().__init__()
        # default 4x 16x 64x 256x

        self.input_dim = 3
        self.num_levels = num_levels
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.concat = concat
        self.output_dim = num_levels * level_dim if concat else level_dim

        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        self.seamless = 1

        params_list = []
        L = float(base_resolution)
        n_elems = 0
        for ii in range(num_levels):
            iL = int(np.ceil(L))
            param = nn.Parameter(torch.empty(6,self.level_dim,iL,iL))
            n_elems += 6 * self.level_dim * iL * iL
            params_list.append(param)
            L = L * per_level_scale
        #self.params_list = nn.ModuleList(*params_list)
        self.params_list = nn.ParameterList(params_list)

        self.fail_value = nn.Parameter(torch.zeros(self.level_dim))
        n_elems += self.level_dim
        self.n_elems = n_elems # total parameters

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        for ii in range(self.num_levels):
            self.params_list[ii].data.uniform_(-std,std)

    def __repr__(self):
        return f"MipCubemapEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} -> {self.n_elems} per_level_scale={self.per_level_scale:.4f} interpolation={self.interpolation} seamless={self.seamless}"

    def forward(self, inputs):
        outputs = []
        for ii in range(self.num_levels):
            x = cubemap_encode(inputs, self.params_list[ii], self.fail_value, self.interp_id, self.seamless)
            outputs.append(x)
        if self.concat:
            outputs = torch.cat(outputs,dim=0)
        else:
            outputs = sum(outputs)
        return outputs.permute(1,0) # CxN -> NxC






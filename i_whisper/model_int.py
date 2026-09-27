import base64
import gzip
import platform
_IS_ARM = platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")
from dataclasses import dataclass
from operator import rshift
from typing import Dict, Iterable, Optional
from logging import Logger
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from bitsandbytes.functional import int8_linear_matmul

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .model_quant import *
from .modules.linear import Linear8, Linear32
from .modules.functional import (
    Rshift,
    Fast_Inverse_Square_Root,
    Batch_Head_s8s8s32_GeMM,
    Batch_Head_u8s8s32_GeMM,
    int_bit_length
)
from i_whisper.kernels import _C

import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union


'''Only Import quantized modules'''
__all__ = [
    "GELU_int",
    "LayerNorm_int",
    "Conv1d_int",
    "MultiHeadAttention_int",
    "ResidualAttentionBlock_int",
    "AudioEncoder_int",
    "TextDecoder_int",
    "Whisper_int"
    ]




@dataclass
class ModelDimensions:
    n_mels: int        # 80
    n_audio_ctx: int   # 1500
    n_audio_state: int # base: 512, tiny: 384
    n_audio_head: int  # base: 8, tiny: 6
    n_audio_layer: int # base: 6, tiny: 4
    n_vocab: int       # 51865
    n_text_ctx: int    # 448
    n_text_state: int  # base: 512, tiny: 384
    n_text_head: int   # base: 8, tiny: 6
    n_text_layer: int  # base: 6, tiny: 4


def Rshift_legacy(x, N):
    if x.dtype is torch.int32:
        ROUNDING_BIAS = (
            1 << (N - 1).to(torch.int32).clamp(0, 32)
        ) - torch.signbit(x).to(torch.int32)

        return (
            x + ROUNDING_BIAS
        ) >> N.to(torch.int32).clamp(0, 32)

    raise NotImplementedError

class GELU_int(nn.Module):
    def __init__(
        self,
        channel_num : int = 1,
        out_channel_num: int = 1,
        backend: str = "torch"
    ) -> None:
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.channel_num = channel_num
        self.backend = backend
        if channel_num == 1:
            self.out_channel_num = out_channel_num
        else:
            self.out_channel_num = channel_num
        # Because ReQuant of multi-channel always need multi-scale
        # So we need to set out_channel_num for multi-channel GELU

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8
        self.lut_index_bit = 4
        self.lut_data_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127# 2^(act_bit-1)-1
        self.lut_upper = 15 # 2^(lut_index_bit)-1
        self.lut_zero = 8  # 2^(lut_data_bit-1)

        '''Quantization Parameters initialization'''
        self.lut = nn.Parameter(torch.arange(0, self.lut_upper+1).to(torch.uint8), requires_grad=False)
        self.luti_req_multiplier = nn.Parameter(torch.ones(channel_num, dtype=torch.uint32), requires_grad=False)
        self.luti_req_shiftmount = nn.Parameter(torch.zeros(channel_num, dtype=torch.int8), requires_grad=False)
        self.luto_req_multiplier = nn.Parameter(torch.ones(out_channel_num, dtype=torch.uint8), requires_grad=False)
        self.luto_req_shiftmount = nn.Parameter(torch.zeros(out_channel_num, dtype=torch.int8), requires_grad=False)
        self.de_smooth_vec = nn.Parameter(torch.ones(channel_num, dtype=torch.uint8), requires_grad=False)
    

    def _forward_torch(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.int8, "Input tensor must be int8 in integer inference mode!"
        # Integer Inference Mode

        index = (x.long().mul(self.luti_req_multiplier.long())>>32).int()
        index = Rshift(index, self.luti_req_shiftmount)
        index = (index+self.lut_zero).clamp(0, self.lut_upper).to(dtype=torch.long, device=x.device)
        index = self.lut[index]

        x = index.int() * x.int()
        x = Rshift(x.mul(self.luto_req_multiplier.int()), self.luto_req_shiftmount)
        x = x.clamp(-self.act_range, self.act_range).to(torch.int8)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.int8, "Input tensor must be int8 in integer inference mode!"
        if self.backend == "torch":
            out = self._forward_torch(x)
        elif self.backend == "scalar":
            out = _C.gelu_scalar(
                x.contiguous(),
                self.lut,
                self.luti_req_multiplier,
                self.luti_req_shiftmount,
                self.luto_req_multiplier,
                self.luto_req_shiftmount
            )
        elif self.backend == "simd":
            out = _C.gelu_simd(
                x.contiguous(),
                self.lut,
                self.luti_req_multiplier,
                self.luti_req_shiftmount,
                self.luto_req_multiplier,
                self.luto_req_shiftmount
            )
        elif self.backend == "check":
            ref = self._forward_torch(x)
            out = _C.gelu_simd(
                x.contiguous(),
                self.lut,
                self.luti_req_multiplier,
                self.luti_req_shiftmount,
                self.luto_req_multiplier,
                self.luto_req_shiftmount
            )
            
            assert torch.equal(out, ref), "GELU mismatch"
        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented.")
        return out
    
    @staticmethod
    def copy_param(quant_layer: 'GELU_quantized') -> None:
        """Copy parameters from quantized layer to new layer
        Args:
            new_layer (GELU_int): target GELU_int layer
            quant_layer (GELU_int): source GELU_int layer
        """
        new_layer = GELU_int(
            channel_num=quant_layer.channel_num,
            out_channel_num=quant_layer.out_channel_num
        )
        new_layer.lut.data.copy_(quant_layer.lut.data)
        new_layer.luti_req_multiplier.data.copy_(quant_layer.luti_req_multiplier.data)
        new_layer.luti_req_shiftmount.data.copy_(quant_layer.luti_req_shiftmount.data)
        new_layer.luto_req_multiplier.data.copy_(quant_layer.luto_req_multiplier.data)
        new_layer.luto_req_shiftmount.data.copy_(quant_layer.luto_req_shiftmount.data)
        new_layer.de_smooth_vec.data.copy_(quant_layer.de_smooth_vec.data)

        return new_layer.to(quant_layer.lut.device)


class LayerNorm_int(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device = "cpu",
        backend: str = "torch",
    ) -> None:
        '''inherit from nn.LayerNorm'''
        super().__init__(normalized_shape, eps, elementwise_affine, device=device)
        
        '''meta parameters'''
        self.backend = backend

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 15
        self.wgt_bit = 8
        self.norm_out_bit = 8
        self.offset = 13

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 16383      # Signed, 2**(self.act_bit-1)-1
        self.wgt_range = 127        # Signed, 2**(self.wgt_bit-1)-1
        self.norm_out_range = 127   # Signed, 2**(self.norm_out_bit-1)-1

        '''Quantization Parameters initialization'''
        self.weight = nn.Parameter(torch.ones_like(self.weight, dtype=torch.int8, device=device), requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.zeros_like(self.bias, dtype=torch.int32, device=device), requires_grad=False)
        self.ln_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8, device=device), requires_grad=False)
        self.ln_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8, device=device), requires_grad=False)
        self.out_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint32, device=device), requires_grad=False)
        self.out_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8, device=device), requires_grad=False)
    
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.int16
        res = x.clone()
        if self.backend == "torch":
            out = self._forward_torch(x)
        elif self.backend == "scalar":
            out = _C.layernorm32_scalar(
                x,
                self.weight,
                self.bias,
                self.ln_req_multiplier,
                self.ln_req_shiftmount,
                self.out_req_multiplier,
                self.out_req_shiftmount,
            )
        elif self.backend == "simd":
            out = _C.layernorm32_simd(
                x,
                self.weight,
                self.bias,
                self.ln_req_multiplier,
                self.ln_req_shiftmount,
                self.out_req_multiplier,
                self.out_req_shiftmount,
            )
        elif self.backend == "check":
            ref = self._forward_torch(x)

            out = _C.layernorm32_simd(
                x,
                self.weight,
                self.bias,
                self.ln_req_multiplier,
                self.ln_req_shiftmount,
                self.out_req_multiplier,
                self.out_req_shiftmount,
            )

            if not torch.equal(out, ref):
                row = 8
                col = 2

                # ----- Torch stats -----
                norm_number = self.normalized_shape[0]

                norm_multiplier = {
                    1280: [True, False, False],
                    1024: [False, False, False],
                    768:  [False, True, True],
                    512:  [False, False, False],
                    384:  [False, True, True],
                }

                norm_shiftmount = {
                    1280: 11,
                    1024: 10,
                    768: 10,
                    512: 9,
                    384: 9,
                }

                def mul_by_shift_multiplier(v, multiplier):
                    o = v.clone()
                    if multiplier[0]:
                        o += v >> 1
                    if multiplier[1]:
                        o += v >> 2
                    if multiplier[2]:
                        o += v >> 3
                    return o

                norm_mul = norm_multiplier[norm_number]
                norm_shift = norm_shiftmount[norm_number]

                mean_ref = torch.sum(
                    x.int(),
                    dim=-1,
                    keepdim=True,
                ).int()

                mean_ref = (
                    mul_by_shift_multiplier(mean_ref, norm_mul)
                    >> norm_shift
                )

                sub_square = (x - mean_ref).abs() ** 2

                max_bit_length = int_bit_length(
                    sub_square.max(
                        dim=-1,
                        keepdim=True,
                    ).values
                )

                renorm_ref = (
                    max_bit_length - 16
                ).clamp(min=0)

                renorm_ref = (
                    renorm_ref + 1
                ) & ~1

                var_ref = torch.sum(
                    Rshift(
                        sub_square,
                        renorm_ref,
                    ),
                    dim=-1,
                    keepdim=True,
                ).int()

                var_ref = (
                    mul_by_shift_multiplier(
                        var_ref,
                        norm_mul,
                    )
                    >> norm_shift
                ).int()

                invstd_ref = Fast_Inverse_Square_Root(
                    var_ref,
                    12,
                    2,
                )

                # ----- C++ debug kernels -----
                mean_cpp, var_cpp, renorm_cpp = (
                    _C.layernorm32_stats_scalar(
                        x.contiguous()
                    )
                )

                flat_row = row  # shape = [1,1500,512]

                invstd_cpp = _C.layernorm32_rsqrt_scalar(
                    var_cpp
                )

                print("==== LN DEBUG ====")
                print(
                    "mean:",
                    mean_ref[0, row, 0].item(),
                    mean_cpp[flat_row].item(),
                )
                print(
                    "renorm:",
                    renorm_ref[0, row, 0].item(),
                    renorm_cpp[flat_row].item(),
                )
                print(
                    "var:",
                    var_ref[0, row, 0].item(),
                    var_cpp[flat_row].item(),
                )
                print(
                    "invstd:",
                    invstd_ref[0, row, 0].item(),
                    invstd_cpp[flat_row].item(),
                )

                print(
                    "input:",
                    x[0, row, col].item(),
                )
                print(
                    "weight:",
                    self.weight[col].item(),
                )
                print(
                    "bias:",
                    self.bias[col].item()
                    if self.bias is not None else None,
                )

                raise RuntimeError("LayerNorm32 mismatch")
        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented.")
        
        return out, res
    


    def _forward_torch(self, x: Tensor) -> Tensor:
        '''LayerNorm with Quantize function
        '''
        norm_number = self.normalized_shape[0]
        # replace norm division to multiplication + right shift
        # 1/1280 = (1/5) * 2^-8 = 1.6 * 2^-11 approx 1.5 * 2^-11
        # 1/1024 = 1 * 2^-10
        # 1/768 = (1/3) * 2^-8 = 1.3333 * 2^-10 approx 1.375 * 2^-10
        # 1/512 = 1 * 2^-9
        # 1/384 = (1/3) * 2^-7 = 1.3333 * 2^-9 approx 1.375 * 2^-9
        norm_multiplier = {
            1280:[True,False,False],  # 'b1.100,  d'1.5
            1024:[False,False,False], # 'b1.000,  d'1.0
            768: [False,True,True],   # 'b1.011,  d'1.375
            512: [False,False,False], # 'b1.000,  d'1.0
            384: [False,True,True]    # 'b1.011,  d'1.375
        }
        def mul_by_shift_multiplier(x, multiplier):
            out = x.clone()
            if multiplier[0]:
                out += (x >> 1) # += multiply by 0.5
            if multiplier[1]:
                out += (x >> 2) # += multiply by 0.25
            if multiplier[2]:
                out += (x >> 3) # += multiply by 0.125
            return out

        norm_shiftmount = {
            1280:11,
            1024:10,
            768:10,
            512:9,
            384:9
        }
        
        norm_mul = norm_multiplier[norm_number]
        norm_shmnt = norm_shiftmount[norm_number]

        mean = torch.sum(x.int(), dim=-1, keepdim=True).int()
        mean = mul_by_shift_multiplier(mean, norm_mul) >> norm_shmnt # reduce by mul & rshift, avoid division
        # print("centered:", (x - mean)[0, row, col].item())
        sub_square = (x - mean).abs()**2

        max_bit_length = int_bit_length(sub_square.max(dim=-1, keepdim=True).values) # max bit usage in sub_square's last dim (hidden dim)
        # OR you can say max clz(count leading zero) in sub_square's last dim
        renorm_shmnt = (max_bit_length - 16).clamp(min=0) # assume 23 bits of every number in sub_square's last dim is enough (fp32 has 23 bits mantissa)
        # this means, we find if anyone is more than 16 bits
        # if true, every number in the group Rshift by the same amount to prevent overflow in the following summation
        renorm_shmnt = (renorm_shmnt + 1) & ~1

        var = torch.sum((Rshift(sub_square, renorm_shmnt)), dim=-1, keepdim=True).int()
        # var += (
        #     (renorm_shmnt == 0).int()
        #     * x.shape[-1]
        # )
        var = (mul_by_shift_multiplier(var, norm_mul) >> norm_shmnt).int() # reduce by mul & rshift, avoid division
        # explicit integer variance stabilization
        # var = var + (renorm_shmnt == 0).to(torch.int32)

        invstd = Fast_Inverse_Square_Root(var, self.offset, 2) # this is in fixed point q20.12 format
        # print("centered*invstd:", ((x - mean) * invstd)[0, row, col].item())
        x = ((x - mean) * invstd)


        '''Fixedpoint Multiplication'''
        # print("*ln_multiplier:", (x * self.ln_req_multiplier)[0, row, col].item())
        # print(
        #     "norm_shift:",
        #      (self.ln_req_shiftmount+(renorm_shmnt>>1))[0, row, 0].item()
        #     if  (self.ln_req_shiftmount+(renorm_shmnt>>1)).dim() == 3
        #     else  (self.ln_req_shiftmount+(renorm_shmnt>>1)).item()
        # )
        x = Rshift( (x * self.ln_req_multiplier), (self.ln_req_shiftmount+(renorm_shmnt>>1)) )
        # print(
        #     "after norm Rshift:",
        #     x[0, row, col].item()
        # )
        x = x.clamp(min=-self.act_range, max=self.act_range)
        # print(
        #     "after clamp:",
        #     x[0, row, col].item()
        # )

        # print(
        #     "after weight:",
        #     (x.mul(self.weight.int()))[0, row, col].item()
        # )
        if self.bias is None:
            x = x.mul(self.weight.int())
        else:
            x = (x.mul(self.weight.int()) + self.bias)
            # print(
            #     "after bias:",
            #     x[0, row, col].item()
            # )

        '''Re-Quantize final output'''
        x = (x.long().mul(self.out_req_multiplier.long())>>32).int()
        # print(
        #     "after MULH:",
        #     x[0, row, col].item()
        # )
        x = Rshift(x, self.out_req_shiftmount)
        # print(
        #     "after final Rshift:",
        #     x[0, row, col].item()
        # )
        x = x.clamp(min=-self.norm_out_range, max=self.norm_out_range).to(torch.int8)

        return x

    @staticmethod
    def copy_param(quant_layer):
        if hasattr(quant_layer, 'quantize') and hasattr(quant_layer, 'int_mode'):
            # this is our self define quantized layernorm
            if quant_layer.quantize and quant_layer.int_mode:
                new_layer = LayerNorm_int(
                    quant_layer.normalized_shape,
                    device=quant_layer.weight.device
                    )
                if new_layer.weight is not None:
                    new_layer.weight.data.copy_(quant_layer.weight.data)
                    new_layer.bias.data.copy_(quant_layer.bias.data)
                new_layer.ln_req_multiplier.data.copy_(quant_layer.ln_req_multiplier.data)
                new_layer.ln_req_shiftmount.data.copy_(quant_layer.ln_req_shiftmount.data)
                new_layer.out_req_multiplier.data.copy_(quant_layer.out_req_multiplier.data)
                new_layer.out_req_shiftmount.data.copy_(quant_layer.out_req_shiftmount.data)
            else:
                print("model not quant yet!")
        else:
            raise NotImplementedError("The input layer is not i-whisper quantize LayerNorm")
        return new_layer.to(quant_layer.weight.device)
        

class Conv1d_int(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        backend: str = "torch"
    ):
        '''inherit from nn.Conv1d'''
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint32), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        self.weight = nn.Parameter(
            torch.ones((self.out_channels, self.in_channels * self.kernel_size[0]), dtype=torch.int8),
            requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.ones((self.out_channels), dtype=torch.int32), requires_grad=False)
        
        self.backend = backend

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        if self.backend == "torch":
            out = self._forward_torch(x, weight, bias)
        elif self.backend == "scalar":
            out = _C.conv1d_scalar(
                x.contiguous(),
                weight.contiguous(),
                bias.contiguous() if bias is not None else None,
                self.req_multiplier,
                self.req_shiftmount,
                self.stride[0],
                self.padding[0]
            )
        elif self.backend == "simd":
            out = _C.conv1d_simd(
                x.contiguous(),
                weight.contiguous(),
                bias.contiguous() if bias is not None else None,
                self.req_multiplier,
                self.req_shiftmount,
                self.stride[0],
                self.padding[0]
            )
        elif self.backend == "check":
            out_ref = self._forward_torch(x, weight, bias)
            out = _C.conv1d_simd(
                x.contiguous(),
                weight.contiguous(),
                bias.contiguous() if bias is not None else None,
                self.req_multiplier,
                self.req_shiftmount,
                self.stride[0],
                self.padding[0]
            )
            assert torch.equal(out, out_ref), "Conv1d mismatch"
        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented.")
        return out
    
    def _forward_torch(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"
        # out = torch.nn.functional.unfold(x.unsqueeze(-1).float(), kernel_size=(self.kernel_size[0],1),
        #     dilation=(self.dilation[0],1), padding=(self.padding[0],0), stride=(self.stride[0],1)).permute(0,2,1).to(torch.int8)
        
        x = torch.nn.functional.pad(
            x,
            (self.padding[0], self.padding[0])
        )

        out = x.unfold(
            dimension=2,
            size=self.kernel_size[0],
            step=self.stride[0]
        )

        out = (
            out
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(
                x.shape[0],
                -1,
                self.in_channels * self.kernel_size[0]
            )
        )
        
        if x.device.type == 'cuda':
            out = int8_linear_matmul(out.contiguous(), self.weight)
        else: # using cpu
            out = torch.matmul(out.int(), self.weight.int().t())
        
        if self.bias is not None:
            out += self.bias
        out = out.permute(0,2,1)

        '''Re-Quantization'''
        out = (out.long().mul(self.req_multiplier.long()) >> 32).int()
        out = Rshift(out, self.req_shiftmount)
        out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)

        return out
    
    @staticmethod
    def copy_param(quant_layer):
        new_layer = Conv1d_int(
            in_channels=quant_layer.in_channels,
            out_channels=quant_layer.out_channels,
            kernel_size=quant_layer.kernel_size[0],
            stride=quant_layer.stride[0],
            padding=quant_layer.padding[0],
            dilation=quant_layer.dilation[0],
            groups=quant_layer.groups,
            bias=quant_layer.bias is not None
        )
        new_layer.weight.data.copy_(quant_layer.weight.data)
        if quant_layer.bias is not None:
            new_layer.bias.data.copy_(quant_layer.bias.data)
        new_layer.req_multiplier.data.copy_(quant_layer.req_multiplier.data)
        new_layer.req_shiftmount.data.copy_(quant_layer.req_shiftmount.data)
        return new_layer.to(quant_layer.weight.device)



class MultiHeadAttention_int(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        backend: str = "torch"
    ):
        '''inherit from nn.Module'''
        super().__init__()
        '''meta parameters'''
        self.n_state = n_state
        self.n_head = n_head
        self.backend = backend
        self.impl = "old"

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.qk_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.qk_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.av_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.av_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.register_buffer(
            "qk_frac_bits",
            torch.tensor(7, dtype=torch.int32)
        )
        self.rsqrt_offset = 16

        '''sub-modules initialization'''
        self.query = Linear8(n_state, n_state)
        self.key = Linear8(n_state, n_state, bias=False)
        self.value = Linear8(n_state, n_state)
        self.out = Linear32(n_state, n_state)
    

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)


    def shift_exp(self,x,row_max,mask):
        x_sub = (x - row_max)
        x_sub[x_sub>0] = torch.iinfo(torch.int16).min
        if mask is not None:
            torch_mask = mask.clone()
            if torch_mask.dim() == 2:
                torch_mask = torch_mask.unsqueeze(0).unsqueeze(0)
            torch_mask = torch_mask.repeat(
                x_sub.shape[0]//torch_mask.shape[0],
                self.n_head//torch_mask.shape[1],
                1,
                1
            )
        else:
            torch_mask = None
        offset = self.qk_frac_bits
        x_mul_fix = x_sub + (x_sub >> 1) - (x_sub >> 4)  # qk_sub_fix * b1.0111
        x_mul_int = x_mul_fix >> offset
        x_mul_frac = x_mul_fix - (x_mul_int << offset)
        x_exp = ((1 << offset) + (x_mul_frac >> 1)) >> (-x_mul_int)
        if torch_mask is not None:
            x_exp[torch_mask] = 0
        x_exp[x_sub>0] = torch.iinfo(torch.int16).min
        if self.qk_frac_bits > 7:
            x_exp = (x_exp >> (self.qk_frac_bits - 7))
        
        x_exp = x_exp.to(torch.uint8)
        x_exp_sum = x_exp.sum(dim=-1, keepdim=True).int()
        return x_exp, x_exp_sum

    
    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        """Prepare attention mask if provided"""
        attn_mask = None

        if mask is not None:
            q_len = q.shape[-2]
            k_len = k.shape[-2]

            q_offset = k_len - q_len

            assert q_offset >= 0

            attn_mask = mask[
                q_offset : q_offset + q_len,
                :k_len
            ].contiguous()

            assert attn_mask.shape == (q_len, k_len)
            
            


        '''Calculate QK^T'''
        qk = None
        qk_row_max = None
        if self.backend == "torch":
            torch_attn_mask = attn_mask.clone() if attn_mask is not None else None
            if torch_attn_mask is not None:
                if torch_attn_mask.dim() == 2:
                    torch_attn_mask = torch_attn_mask.unsqueeze(0).unsqueeze(0)
                torch_attn_mask = torch_attn_mask.repeat(
                    q.shape[0]//torch_attn_mask.shape[0],
                    self.n_head//torch_attn_mask.shape[1],
                    1,
                    1
                )

            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = Batch_Head_s8s8s32_GeMM(q, k)  # int32 tensor
            qk = Rshift(qk.mul(self.qk_req_multiplier.int()), self.qk_req_shiftmount)
            qk = qk.clamp(torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max).to(torch.int16)
        
            if torch_attn_mask is not None:
                qk[torch_attn_mask] = torch.iinfo(torch.int16).min
            qk_row_max = qk.max(dim=-1, keepdim=True).values.to(torch.int16) + 1
            
        elif self.backend == "scalar":
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            qk, qk_row_max = _C.qk_requant_rowmax_scalar(
                q.contiguous(),
                k.contiguous(),
                attn_mask,
                self.qk_req_multiplier,
                self.qk_req_shiftmount
            )
        
        elif self.backend == "simd":
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            qk, qk_row_max = _C.qk_requant_rowmax_simd(
                q.contiguous(),
                k.contiguous(),
                attn_mask,
                self.qk_req_multiplier,
                self.qk_req_shiftmount
            )
        
        elif self.backend == "check":
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk, qk_row_max = _C.qk_requant_rowmax_simd(
                q.contiguous(),
                k.contiguous(),
                attn_mask,
                self.qk_req_multiplier,
                self.qk_req_shiftmount
            )

            torch_attn_mask = attn_mask.clone() if attn_mask is not None else None
            if torch_attn_mask is not None:
                if torch_attn_mask.dim() == 2:
                    torch_attn_mask = torch_attn_mask.unsqueeze(0).unsqueeze(0)
                torch_attn_mask = torch_attn_mask.repeat(
                    q.shape[0]//torch_attn_mask.shape[0],
                    self.n_head//torch_attn_mask.shape[1],
                    1,
                    1
                )

            ref_qk = Batch_Head_s8s8s32_GeMM(q, k)  # int32 tensor
            ref_qk = Rshift(ref_qk.mul(self.qk_req_multiplier.int()), self.qk_req_shiftmount)
            ref_qk = ref_qk.clamp(torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max).to(torch.int16)
        
            if torch_attn_mask is not None:
                ref_qk[torch_attn_mask] = torch.iinfo(torch.int16).min
            ref_qk_row_max = ref_qk.max(dim=-1, keepdim=True).values.to(torch.int16) + 1

            
            assert torch.equal(qk, ref_qk), "QK mismatch"
            assert torch.equal(qk_row_max, ref_qk_row_max), "QK row max mismatch"

        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented.")

        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)

        if self.backend == "torch":
            qk_exp, qk_exp_sum = self.shift_exp(
                qk,
                qk_row_max,
                attn_mask,
            )
        elif self.backend == "scalar":
            qk_exp, qk_exp_sum = _C.shift_exp_scalar(
                qk,
                qk_row_max,
                attn_mask,
                self.qk_frac_bits,
            )
        elif self.backend == "simd":
            qk_exp, qk_exp_sum = _C.shift_exp_simd(
                qk,
                qk_row_max,
                attn_mask,
                self.qk_frac_bits,
            )
        elif self.backend == "check":
            qk_exp_ref, qk_exp_ref_sum = self.shift_exp(
                qk,
                qk_row_max,
                attn_mask,
            )
            qk_exp, qk_exp_sum = _C.shift_exp_simd(
                qk,
                qk_row_max,
                attn_mask,
                self.qk_frac_bits,
            )
            
            assert torch.equal(qk_exp, qk_exp_ref), "QK exp mismatch"
            assert torch.equal(qk_exp_sum, qk_exp_ref_sum), "QK exp sum mismatch"
        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented.")
        
        if self.backend == "torch":
            out = Batch_Head_u8s8s32_GeMM(qk_exp, v)  # int32 tensor
            sum_inv_mul = Fast_Inverse_Square_Root(qk_exp_sum, self.rsqrt_offset, Newton_iter=2)
            out = Rshift(out.mul(sum_inv_mul).int(), self.rsqrt_offset)
            out = Rshift(out.mul(sum_inv_mul).int(), self.rsqrt_offset)
            out = Rshift(out.mul(self.av_req_multiplier.int()).int(), self.av_req_shiftmount)
            out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
            out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
        elif self.backend == "scalar":
            out = _C.pv_norm_requant_scalar(
                qk_exp.contiguous(),
                v.contiguous(),
                qk_exp_sum.contiguous(),
                self.av_req_multiplier,
                self.av_req_shiftmount,
                self.rsqrt_offset,
            )
        elif self.backend == "simd":
            out = _C.pv_norm_requant_simd(
                qk_exp.contiguous(),
                v.contiguous(),
                qk_exp_sum.contiguous(),
                self.av_req_multiplier,
                self.av_req_shiftmount,
                self.rsqrt_offset,
            )
        elif self.backend == "check":
            out_ref = Batch_Head_u8s8s32_GeMM(qk_exp, v)  # int32 tensor
            sum_inv_mul_ref = Fast_Inverse_Square_Root(qk_exp_sum, self.rsqrt_offset, Newton_iter=2)
            out_ref = Rshift(out_ref.mul(sum_inv_mul_ref).int(), self.rsqrt_offset)
            out_ref = Rshift(out_ref.mul(sum_inv_mul_ref).int(), self.rsqrt_offset)
            out_ref = Rshift(out_ref.mul(self.av_req_multiplier.int()).int(), self.av_req_shiftmount)
            out_ref = out_ref.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
            out_ref = out_ref.permute(0, 2, 1, 3).flatten(start_dim=2)

            out = _C.pv_norm_requant_simd(
                qk_exp.contiguous(),
                v.contiguous(),
                qk_exp_sum.contiguous(),
                self.av_req_multiplier,
                self.av_req_shiftmount,
                self.rsqrt_offset,
            )
            assert torch.equal(out, out_ref), "PV normalize/requant mismatch"
        else:
            raise NotImplementedError(f"Backend {self.backend} is not implemented.")

        return out
    
    @staticmethod
    def copy_param(quant_layer):
        new_layer = MultiHeadAttention_int(
            n_state=quant_layer.n_state,
            n_head=quant_layer.n_head
        )
        new_layer.qk_req_multiplier.data.copy_(quant_layer.qk_req_multiplier.data)
        new_layer.qk_req_shiftmount.data.copy_(quant_layer.qk_req_shiftmount.data)
        new_layer.av_req_multiplier.data.copy_(quant_layer.av_req_multiplier.data)
        new_layer.av_req_shiftmount.data.copy_(quant_layer.av_req_shiftmount.data)
        new_layer.qk_frac_bits.data.copy_(quant_layer.qk_frac_bits.data)
        new_layer.query = Linear8.copy_param(quant_layer.query)
        new_layer.key = Linear8.copy_param(quant_layer.key)
        new_layer.value = Linear8.copy_param(quant_layer.value)
        new_layer.out = Linear32.copy_param(quant_layer.out)

        return new_layer.to(quant_layer.qk_req_multiplier.device)



class ResidualAttentionBlock_int(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        backend = "torch"
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.backend = backend

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_attn_bit = 15
        self.act_cross_attn_bit = 15
        self.act_mlp_bit = 15

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_attn_range = 16383         # Signed
        self.act_cross_attn_range = 16383   # Signed
        self.act_mlp_range = 16383          # Signed

        '''Quantization Parameters initialization'''
        self.attn_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint16), requires_grad=False)
        self.cross_attn_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint16), requires_grad=False)
        self.mlp_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint16), requires_grad=False)

        '''sub-modules initialization'''
        # Attention Module
        self.attn = MultiHeadAttention_int(n_state, n_head)
        self.attn_ln = LayerNorm_int(n_state)

        # Cross-Attention Module (if needed)
        self.cross_attn = MultiHeadAttention_int(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm_int(n_state) if cross_attention else None

        # MLP Module
        self.n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear8(n_state, self.n_mlp),
            GELU_int(channel_num = self.n_mlp, out_channel_num = self.n_mlp),
            Linear32(self.n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm_int(n_state)


    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
        module_inout = self.attn_ln(x)[0]

        '''Compute Attention'''
        module_inout = self.attn(module_inout, mask=mask, kv_cache=kv_cache)

        '''Residual connection with/without quantization'''
        if self.backend == "torch":
            x = self._residual15_torch(module_inout, x, self.attn_res_req_multiplier)
        elif self.backend == "simd":
            x = _C.residual15_simd(module_inout, x, self.attn_res_req_multiplier)
        elif self.backend == "scalar":
            x = _C.residual15_scalar(module_inout, x, self.attn_res_req_multiplier)
        elif self.backend == "check":
            x_check = self._residual15_torch(module_inout, x, self.attn_res_req_multiplier)
            x = _C.residual15_simd(module_inout, x, self.attn_res_req_multiplier)
            assert torch.equal(x, x_check), "Residual connection mismatch between simd and torch!"


        '''if exists Cross-Attention Module, compute Cross-Attention'''
        if self.cross_attn:

            '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
            module_inout = self.cross_attn_ln(x)[0]

            '''Compute Cross-Attention, because cross-attention so need xa input (xa stands for x audio)'''
            module_inout = self.cross_attn(module_inout, xa, kv_cache=kv_cache)

            '''Residual connection with/without quantization'''
            if self.backend == "torch":
                x = self._residual15_torch(module_inout, x, self.cross_attn_res_req_multiplier)
            elif self.backend == "simd":
                x = _C.residual15_simd(module_inout, x, self.cross_attn_res_req_multiplier)
            elif self.backend == "scalar":
                x = _C.residual15_scalar(module_inout, x, self.cross_attn_res_req_multiplier)
            elif self.backend == "check":
                x_check = self._residual15_torch(module_inout, x, self.cross_attn_res_req_multiplier)
                x = _C.residual15_simd(module_inout, x, self.cross_attn_res_req_multiplier)
                assert torch.equal(x, x_check), "Residual connection mismatch between simd and torch!"

        '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
        module_inout = self.mlp_ln(x)[0]
        
        '''Compute MLP'''
        module_inout = self.mlp(module_inout)
        
        '''Residual connection with/without quantization'''
        if self.backend == "torch":
            x = self._residual15_torch(module_inout, x, self.mlp_res_req_multiplier)
        elif self.backend == "simd":
            x = _C.residual15_simd(module_inout, x, self.mlp_res_req_multiplier)
        elif self.backend == "scalar":
            x = _C.residual15_scalar(module_inout, x, self.mlp_res_req_multiplier)
        elif self.backend == "check":
            x_check = self._residual15_torch(module_inout, x, self.mlp_res_req_multiplier)
            x = _C.residual15_simd(module_inout, x, self.mlp_res_req_multiplier)
            assert torch.equal(x, x_check), "Residual connection mismatch between simd and torch!"

        return x
    
    
    def _residual15_torch(self, branch: Tensor, residual: Tensor, req_multiplier: Tensor) -> Tensor:
        assert branch.dtype == torch.int32, "branch tensor must be int32 in integer inference mode!"
        assert residual.dtype == torch.int16, "residual tensor must be int16 in integer inference mode!"
        residual = residual.int().mul(req_multiplier.int())
        branch = Rshift(branch + residual, 15)
        return branch.clamp(-16383, 16383).to(torch.int16)
    
    
    
    @staticmethod
    def copy_param(quant_layer):
        new_layer = ResidualAttentionBlock_int(
            n_state=quant_layer.attn_ln.normalized_shape[0],
            n_head=quant_layer.attn.n_head,
            cross_attention=quant_layer.cross_attn is not None
        )
        new_layer.attn_res_req_multiplier.data.copy_(quant_layer.attn_res_req_multiplier.data)
        new_layer.cross_attn_res_req_multiplier.data.copy_(quant_layer.cross_attn_res_req_multiplier.data)
        new_layer.mlp_res_req_multiplier.data.copy_(quant_layer.mlp_res_req_multiplier.data)
        new_layer.attn = MultiHeadAttention_int.copy_param(quant_layer.attn)
        new_layer.attn_ln = LayerNorm_int.copy_param(quant_layer.attn_ln)
        if quant_layer.cross_attn is not None:
            new_layer.cross_attn = MultiHeadAttention_int.copy_param(quant_layer.cross_attn)
            new_layer.cross_attn_ln = LayerNorm_int.copy_param(quant_layer.cross_attn_ln)
        new_layer.mlp = nn.Sequential(
            Linear8.copy_param(quant_layer.mlp[0]),
            GELU_int.copy_param(quant_layer.mlp[1]),
            Linear32.copy_param(quant_layer.mlp[2])
        )
        new_layer.mlp_ln = LayerNorm_int.copy_param(quant_layer.mlp_ln)

        return new_layer.to(quant_layer.attn_res_req_multiplier.device)



class AudioEncoder_int(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        backend: str = "torch"
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.backend = backend

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_pe_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_pe_range = 127    # Signed, 2**(self.act_pe_bit-1)-1
        self.act_range = 16383     # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.req_multiplier = nn.Parameter(torch.ones((n_state), dtype=torch.uint16), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros((n_state), dtype=torch.int8), requires_grad=False)

        '''sub-modules initialization'''
        self.register_buffer("positional_embedding", torch.empty((n_ctx, n_state),dtype=torch.int8))
        self.input_alpha = nn.Parameter( torch.ones(1) )
        self.gelu1 = GELU_int()
        self.gelu2 = GELU_int(out_channel_num=self.n_state)
        self.conv1 = Conv1d_int(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d_int(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.blocks: Iterable[ResidualAttentionBlock_int] = nn.ModuleList(
            [ResidualAttentionBlock_int(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm_int(n_state)
    
    
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        if x.dtype != torch.int8:
            # quantize input mel spectrogram to int8
            x = (x * self.input_alpha).round().clamp(min=-127, max=127).to(torch.int8)

        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.gelu2(x)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
       
        # x is already quantized in conv layers
        # add positional embedding ( offline quantized )
        if self.backend == "torch":
            x = (x.int() + self.positional_embedding)
            x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount)
            x = x.clamp(min=-self.act_range, max=self.act_range).to(torch.int16)
        elif self.backend == "scalar":
            x = _C.add_requant15_scalar(
                x.contiguous(),
                self.positional_embedding.contiguous(),
                self.req_multiplier,
                self.req_shiftmount
            )
        elif self.backend == "simd":
            x = _C.add_requant15_simd(
                x.contiguous(),
                self.positional_embedding.contiguous(),
                self.req_multiplier,
                self.req_shiftmount
            )
        elif self.backend == "check":
            x_check = (x.int() + self.positional_embedding)
            x_check = Rshift(x_check.mul(self.req_multiplier.int()), self.req_shiftmount)
            x_check = x_check.clamp(min=-self.act_range, max=self.act_range).to(torch.int16)
            x = _C.add_requant15_simd(
                x.contiguous(),
                self.positional_embedding.contiguous(),
                self.req_multiplier,
                self.req_shiftmount
            )
            assert torch.equal(x, x_check), "Add/Requant mismatch between simd and torch!"
        
        '''Pass through Transformer Blocks'''
        for _ in range(self.n_layer):
            x = self.blocks[_](x)

        '''Compute Final LayerNorm'''
        x = self.ln_post(x)[0]

        return x
    
    
    @staticmethod
    def copy_param(quant_layer):
        new_layer = AudioEncoder_int(
            n_mels=quant_layer.n_mels,
            n_ctx=quant_layer.n_ctx,
            n_state=quant_layer.n_state,
            n_head=quant_layer.n_head,
            n_layer=quant_layer.n_layer
        )
        new_layer.req_multiplier.data.copy_(quant_layer.req_multiplier.data)
        new_layer.req_shiftmount.data.copy_(quant_layer.req_shiftmount.data)
        new_layer.positional_embedding.data.copy_(quant_layer.positional_embedding.data)
        new_layer.input_alpha.data.copy_(quant_layer.input_alpha.data)
        new_layer.gelu1 = GELU_int.copy_param(quant_layer.gelu1)
        new_layer.gelu2 = GELU_int.copy_param(quant_layer.gelu2)
        new_layer.conv1 = Conv1d_int.copy_param(quant_layer.conv1)
        new_layer.conv2 = Conv1d_int.copy_param(quant_layer.conv2)
        for i in range(quant_layer.n_layer):
            new_layer.blocks[i] = ResidualAttentionBlock_int.copy_param(quant_layer.blocks[i])
        new_layer.ln_post = LayerNorm_int.copy_param(quant_layer.ln_post)

        return new_layer.to(quant_layer.req_multiplier.device)



class TextDecoder_int(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        backend: str = "torch"
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.backend = backend

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.dequant_need = True

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.logit_range = 127   # Signed, 2**(self.act_bit-1)-1
        self.act_range = 16383   # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.req_multiplier = nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.clf_req_multiplier = torch.nn.Parameter(torch.ones(1, dtype=torch.uint32), requires_grad=False)
        self.clf_req_shiftmount = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.act_logits_alpha = torch.nn.Parameter(torch.ones(1))

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.token_embedding.weight = nn.Parameter(
            self.token_embedding.weight.to(torch.int8).contiguous(),
            requires_grad=False
        )
        self.positional_embedding = nn.Parameter(torch.empty((n_ctx, n_state), dtype=torch.int8), requires_grad=False)

        self.blocks: Iterable[ResidualAttentionBlock_int] = nn.ModuleList(
            [
                ResidualAttentionBlock_int(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm_int(n_state)

        mask = torch.ones((n_ctx, n_ctx), dtype=torch.bool).triu_(1).to(torch.bool)
        self.register_buffer("mask", mask, persistent=False)
    

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """

        '''Calculate offset, offset is the length of cached key/value pairs in kv_cache'''
        '''offset is used to index positional embeddings correctly when using kv_cache'''
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0

        '''Calculate token embeddings with/without quantization'''
        if self.backend == "torch":
            x = ( self.token_embedding(x).to(torch.int32) + self.positional_embedding[offset : offset + x.shape[-1]].to(torch.int32) )
            x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount)
            x = x.clamp(min=-self.act_range, max=self.act_range).to(torch.int16)
        elif self.backend == "scalar":
            x = _C.add_requant15_scalar(
                self.token_embedding(x).contiguous(),
                self.positional_embedding[offset : offset + x.shape[-1]].contiguous(),
                self.req_multiplier.to(torch.uint16),
                self.req_shiftmount
            )
        elif self.backend == "simd":
            x = _C.add_requant15_simd(
                self.token_embedding(x).contiguous(),
                self.positional_embedding[offset : offset + x.shape[-1]].contiguous(),
                self.req_multiplier.to(torch.uint16),
                self.req_shiftmount
            )
        elif self.backend == "check":
            x_check = ( self.token_embedding(x).to(torch.int32) + self.positional_embedding[offset : offset + x.shape[-1]].to(torch.int32) )
            x_check = Rshift(x_check.mul(self.req_multiplier.int()), self.req_shiftmount)
            x_check = x_check.clamp(min=-self.act_range, max=self.act_range).to(torch.int16)
            x = _C.add_requant15_simd(
                self.token_embedding(x).contiguous(),
                self.positional_embedding[offset : offset + x.shape[-1]].contiguous(),
                self.req_multiplier.to(torch.uint16),
                self.req_shiftmount
            )
            assert torch.equal(x, x_check), "Add/Requant mismatch between simd and torch!"

        for _ in range(self.n_layer):
            x = self.blocks[_](x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)[0]

        if self.backend == "torch":
            # int8 matmul with token embedding weight
            if torch.cuda.is_available(): # if GPU is available, use faster int8 GEMM
                x = int8_linear_matmul(x, self.token_embedding.weight)
            else:
                x = torch.matmul(x.int(), self.token_embedding.weight.int().t())
            # requant logits
            x = (x.long().mul(self.clf_req_multiplier.long())>>32).int()
            x = Rshift(x, self.clf_req_shiftmount)
            x = x.clamp(min=-self.logit_range, max=self.logit_range).to(torch.int8)
        elif self.backend == "scalar":
            x = _C.classifier_scalar(
                x.contiguous(),
                self.token_embedding.weight.contiguous(),
                self.clf_req_multiplier,
                self.clf_req_shiftmount
            )
        elif self.backend == "simd":
            x = _C.classifier_simd(
                x.contiguous(),
                self.token_embedding.weight.contiguous(),
                self.clf_req_multiplier,
                self.clf_req_shiftmount
            )
        elif self.backend == "check":
            x_check = int8_linear_matmul(x, self.token_embedding.weight)
            x_check = (x_check.long().mul(self.clf_req_multiplier.long())>>32).int()
            x_check = Rshift(x_check, self.clf_req_shiftmount)
            x_check = x_check.clamp(min=-self.logit_range, max=self.logit_range).to(torch.int8)
            x = _C.classifier_scalar(
                x.contiguous(),
                self.token_embedding.weight.contiguous(),
                self.clf_req_multiplier,
                self.clf_req_shiftmount
            )
            assert torch.equal(x, x_check), "Logits mismatch between simd and torch!"
        if self.dequant_need:
            x = x.float().div(self.act_logits_alpha)
        return x
    
    
    @staticmethod
    def copy_param(quant_layer):
        new_layer = TextDecoder_int(
            n_vocab=quant_layer.n_vocab,
            n_ctx=quant_layer.n_ctx,
            n_state=quant_layer.n_state,
            n_head=quant_layer.n_head,
            n_layer=quant_layer.n_layer
        )
        new_layer.req_multiplier.data.copy_(quant_layer.req_multiplier.data)
        new_layer.req_shiftmount.data.copy_(quant_layer.req_shiftmount.data)
        new_layer.clf_req_multiplier.data.copy_(quant_layer.clf_req_multiplier.data)
        new_layer.clf_req_shiftmount.data.copy_(quant_layer.clf_req_shiftmount.data)
        new_layer.act_logits_alpha.data.copy_(quant_layer.act_logits_alpha.data)
        new_layer.token_embedding.weight.data.copy_(quant_layer.token_embedding.weight.data)
        new_layer.positional_embedding.data.copy_(quant_layer.positional_embedding.data)
        for i in range(quant_layer.n_layer):
            new_layer.blocks[i] = ResidualAttentionBlock_int.copy_param(quant_layer.blocks[i])
        new_layer.ln = LayerNorm_int.copy_param(quant_layer.ln)
        new_layer.mask.data.copy_(quant_layer.mask.data)

        return new_layer.to(quant_layer.req_multiplier.device)



QUANT_MODULES = [
    Conv1d_int,
    GELU_int,
    LayerNorm_int,
    AudioEncoder_int,
    TextDecoder_int,
    ResidualAttentionBlock_int,
    MultiHeadAttention_int
]



class Whisper_int(nn.Module):
    def __init__(self, dims: ModelDimensions):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.dims = dims

        '''sub-modules initialization'''
        self.encoder = AudioEncoder_int(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder_int(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention_int):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function



def set_residual_backend(
    model: Whisper_int,
    backend: str,
    logger: Optional[Logger] = None,
):
    """
    Set the backend for residual connections in the model.
    """
    assert backend in ["torch", "simd", "scalar", "check"], \
        f"Invalid backend: {backend}"

    log = logger.debug if logger is not None else print

    for name, layer in model.named_modules():
        if isinstance(layer, ResidualAttentionBlock_int):
            if layer.backend == backend:
                log(
                    f"Residual backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"Residual backend for {name} "
                    f"from {layer.backend} to {backend}."
                )


def set_layernorm_backend(
    model,
    backend,
    logger=None,
):
    assert backend in [
        "torch",
        "scalar",
        "simd",
        "check",
    ]

    log = (
        logger.debug
        if logger is not None
        else print
    )

    for name, layer in model.named_modules():
        if isinstance(layer, LayerNorm_int):
            if layer.backend == backend:
                log(
                    f"LayerNorm backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"LayerNorm backend for {name} "
                    f"set to {backend}."
                )
            
            # if layer.offset == 12:
            #     print(f"LayerNorm offset for {name} is 12, changing to 13.")
            #     layer.offset = 13
            #     layer.ln_req_shiftmount.data.copy_(layer.ln_req_shiftmount.data + 1)
            #     print(f"LayerNorm ln_req_shiftmount for {name} changed to {layer.ln_req_shiftmount.data.item()}.")


def set_gelu_backend(
    model,
    backend,
    logger=None,
):
    assert backend in [
        "torch",
        "scalar",
        "simd",
        "check",
    ]

    log = (
        logger.debug
        if logger is not None
        else print
    )

    for name, layer in model.named_modules():
        if isinstance(layer, GELU_int):
            if layer.backend == backend:
                log(
                    f"GELU backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"GELU backend for {name} "
                    f"set to {backend}."
                )


def set_attention_backend(
    model,
    backend,
    logger=None,
):
    assert backend in [
        "torch",
        "scalar",
        "simd",
        "check",
    ]

    log = (
        logger.debug
        if logger is not None
        else print
    )

    for name, layer in model.named_modules():
        if isinstance(layer, MultiHeadAttention_int):
            if layer.backend == backend:
                log(
                    f"MultiHeadAttention backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"MultiHeadAttention backend for {name} "
                    f"set to {backend}."
                )

def set_conv1d_backend(
    model,
    backend,
    logger=None,
):
    assert backend in [
        "torch",
        "scalar",
        "simd",
        "check",
    ]

    log = (
        logger.debug
        if logger is not None
        else print
    )

    for name, layer in model.named_modules():
        if isinstance(layer, Conv1d_int):
            if layer.backend == backend:
                log(
                    f"Conv1d backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"Conv1d backend for {name} "
                    f"set to {backend}."
                )

def set_audio_encoder_backend(
    model,
    backend,
    logger=None,
):
    assert backend in [
        "torch",
        "scalar",
        "simd",
        "check",
    ]

    log = (
        logger.debug
        if logger is not None
        else print
    )

    for name, layer in model.named_modules():
        if isinstance(layer, AudioEncoder_int):
            if layer.backend == backend:
                log(
                    f"AudioEncoder backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"AudioEncoder backend for {name} "
                    f"set to {backend}."
                )

def set_text_decoder_backend(
    model,
    backend,
    logger=None,
):
    assert backend in [
        "torch",
        "scalar",
        "simd",
        "check",
    ]

    log = (
        logger.debug
        if logger is not None
        else print
    )

    for name, layer in model.named_modules():
        if isinstance(layer, TextDecoder_int):
            if layer.backend == backend:
                log(
                    f"TextDecoder backend for {name} "
                    f"is already set to {backend}."
                )
            else:
                layer.backend = backend
                log(
                    f"TextDecoder backend for {name} "
                    f"set to {backend}."
                )
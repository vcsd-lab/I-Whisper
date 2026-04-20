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
from .kernels import w8a8

import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List, Optional, Tuple, Union


'''Only Import quantized modules'''
__all__ = [
    "GELU_int",
    "LayerNorm_int",
    "Linear_int",
    "Conv1d_int",
    "MultiHeadAttention_int",
    "ResidualAttentionBlock_int",
    "AudioEncoder_int",
    "TextDecoder_int",
    "Whisper_int",
    "Build_from_Quant"
    ]


def Rshift(x:torch.Tensor, N):
    """Right Shift operation with Rounding-to-Nearest
    """
    '''
    PrePare a number add to result before right shift to realize rounding
    for q2.3 round to q2.3 as example:

    NOTE:POSITIVE EXAMPLE:
    =========================================================================
      0.375 = 'b00.011 -> 0 = 'b00.000  ||  0.875 = 'b00.111 -> 1 = 'b01.000
      ADD 0.5 'b00.100    ^             || ADD 0.5  'b00.100    ^
      ----------------    |             || -----------------    |
      0.875 = 'b00.111    |             ||  1.375 = 'b01.011    |
      MASK(RSHIFT3)XXX    |             || MASK(RSHIFT3) XXX    |
      ----------------    |             || -----------------    |
      0.000 = 'b00.000 -> 0             ||  1.000 = 'b01.000 -> 1
    =========================================================================
    NOTE:NEGATIVE EXAMPLE:                                                 
      -0.375 = 'b11.101 -> 0 = 'b00.000 || -0.675 = 'b11.011 -> -1 = 'b11.000
      ADD 0.375'b00.011    ^            || ADD 0.375'b00.011     ^
      -----------------    |            || -----------------     |
       0.000 = 'b00.000    |            || -0.500 = 'b11.100     |
      MASK(RSHIFT3) XXX    |            || MASK(RSHIFT3) XXX     |
      -----------------    |            || -----------------     |
       0.000 = 'b00.000 -> 0            || -1.000 = 'b11.000 -> -1
    =========================================================================
    Conclusion:
        for QM.N (Signed) Fixpoint Integer rounding, we need to add:
        for POSITIVE number: 2^(N-1)
        for NEGATIVE number: 2^(N-1) - 1
        So the rounding bias is: 2^(N-1) - signbit
    '''
    assert x.dtype is torch.int32, "Rshift function only support int32 tensor!"
    if _IS_ARM:
        return w8a8.Rshift(x, N)
    else:
        try:
            ROUNDING_BIAS = (1 << (N-1).to(torch.int32).clamp(-32,32)) - torch.signbit(x).int()
            return (x + ROUNDING_BIAS)>>N.to(torch.int32).clamp(-32,32)
        except:
            ROUNDING_BIAS = (1 << (N-1)) - torch.signbit(x).int()
            return (x + ROUNDING_BIAS)>>N



def Batch_Head_s8s8s32_GeMM(A: Tensor, B: Tensor):
    """Batch head matrix multiplication for quantized tensors
    Args:
        A (Tensor): [batch, n_head, n_ctx, n_state] int8 tensor (column-major)
        B (Tensor): [batch, n_head, n_ctx, n_state] int8 tensor (row-major)
    Returns:
        Tensor: [batch, n_head, n_ctx, n_ctx] int32 tensor
    """
    Out = torch.zeros((A.shape[0], A.shape[1], A.shape[2], B.shape[2]), dtype=torch.int32, device=A.device)
    if A.device.type == 'cuda':
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                Out[i][j] = int8_linear_matmul(A[i][j].contiguous(), B[i][j].contiguous())
    else:
        if _IS_ARM:
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    Out[i][j] = w8a8.s8s8s32_gemm(A[i][j].contiguous(), B[i][j].contiguous())
        else:
            Out = A.to('cpu',torch.int32) @ B.to('cpu',torch.int32).transpose(-1,-2)
    return Out


def Batch_Head_u8s8s32_GeMM(A: Tensor, B: Tensor):
    """Batch head matrix multiplication for quantized tensors
    Args:
        A (Tensor): [batch, n_head, n_ctx, n_state] uint8 tensor (row-major)
        B (Tensor): [batch, n_head, n_ctx, n_state] int8 tensor (column-major)
    Returns:
        Tensor: [batch, n_head, n_ctx, n_ctx] int32 tensor
    """
    Out = torch.zeros((A.shape[0], A.shape[1], A.shape[2], B.shape[2]), dtype=torch.int32, device=A.device)
    if A.device.type == 'cuda':
        # shift uint8 to int8, then use int8 GEMM, finally shift zero point back
        A_shift = A.to(torch.int8) - 128
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                Out[i][j] = int8_linear_matmul(A_shift[i][j].contiguous(), B[i][j].contiguous())
                Out[i][j] += (torch.sum(B[i][j].to(torch.int32), dim=1, keepdim=True) << 7).t()
    else:
        if _IS_ARM:
            for i in range(A.shape[0]):
                for j in range(B.shape[1]):
                    Out[i][j] = w8a8.u8s8s32_gemm(A[i][j].contiguous(), B[i][j].contiguous())
        else:
            Out = A.to('cpu',torch.int32) @ B.to('cpu',torch.int32).transpose(-1,-2)
    return Out



def Fast_Inverse_Square_Root(x, offset, Newton_iter: int = 0, overflow_prevent_offset: int = 0):
    '''TODO: Very critical problem
    The Newton's method iteration in Fixpoint q12.20 (or too much fraction) format
    may overflow when using int32 format. And if using less decimal bits
    (like q8.24), the precision is not enough to get good accuracy. This Problem can
    be solved by using int64 format, but the speed and support of hardware will be
    a new problem.
    It is a trade-off between supportment and useable inference.
    2025.11.13 Chen96013
    '''
    '''fast inverse square root in flaot32 format'''
    zero_pos = (x == 0)
    i = x.type(torch.float32).view(dtype=torch.int32)
    i = int(0x5f3759df) - (i >> 1) # initial point with magic number

    '''convert potential float32 in int32 to fixed point q12.20 format'''
    # try new convert method
    # add offset to exponent directly then convert to fixed point (int32)
    y = (i + (offset<<23))
    y = y.view(torch.float32)
    y = y.to(torch.int32)
    # without this, inverse square root of zero is infinity, here we set it to one directly
    # this means do not need normalize when input (variation) is zero

    
    '''iteration of Newton's method (in fixed point q12.20 format)'''
    three = torch.tensor(3*2**offset, dtype=torch.int32) # 3 in fixed point q12.20 format
    # threehalfs = 0x0030_0000
    # NOTE: actually threehalfs = 1.5 * (2**20) = 0x0018_0000
    # but we store 3*2**20
    # this is because in Newton's method, we have 1.5 - 0.5 * t
    # we can do (3 - t)>>1 to preserve more precision

    # NOTE: @overflow_prvent_offset this is a hack to prevent overflow during multiplication
    # RShift both y and threehalfs-t by overflow_prevent_offset
    # so that the multiplication result won't overflow int32
    # and the final result is LShifted back by overflow_prevent_offset
    # the RShift is represented as RShift lesser when point regression
    # A * B >> offset
    #          ^ this is point regression, recover the point by right shifting offset bits

    def Newton_iteration(y, x, three, offset):
        t = Rshift(x.mul(y).int(),offset).mul(y)
        # NOTE: here we do not need Rshmnt because var is q32.0 (integer), y2 is q12.20
        # first iteration of Newton's method
        y = Rshift(y.mul( three - t ).int(), (offset+1))
        # this result is in q12.20 format, we can directly use it for next iteration or return it
        return y

    # Do Newton iterations Newton_iter times
    for i in range(Newton_iter):
        y = Newton_iteration(y, x, three, offset)

    return y


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


class GELU_int(nn.Module):
    def __init__(
        self,
        approximate: str = 'none',
        channel_num : int = 1,
        out_channel_num : int = 1,
        debug: bool = False,
    ) -> None:
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.approximate = approximate
        self.channel_num = channel_num
        self.out_channel_num = out_channel_num
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8
        self.lut_index_bit = 4
        self.lut_data_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127# 2^(act_bit-1)-1
        self.lut_upper = 15 # 2^(lut_index_bit)-1

        '''Quantization Parameters initialization'''
        self.lut = nn.Parameter(torch.arange(0, self.lut_upper+1).to(torch.uint8), requires_grad=False)
        self.luti_req_multiplier = nn.Parameter(torch.ones(channel_num, dtype=torch.uint8), requires_grad=False)
        self.luti_req_shiftmount = nn.Parameter(torch.zeros(channel_num, dtype=torch.int8), requires_grad=False)
        self.luto_req_multiplier = nn.Parameter(torch.ones(out_channel_num, dtype=torch.uint8), requires_grad=False)
        self.luto_req_shiftmount = nn.Parameter(torch.zeros(out_channel_num, dtype=torch.int8), requires_grad=False)
        self.de_smooth_vec = nn.Parameter(torch.ones(self.channel_num, dtype=torch.uint8), requires_grad=False)
    

    def forward(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.int8, "Input tensor must be int8 in integer inference mode!"
        # Integer Inference Mode

        '''De-Smoothing and LUT lookup (with Quantize)'''
        index = x.int() * self.de_smooth_vec.int()
        index = Rshift(index.mul(self.luti_req_multiplier.int()), self.luti_req_shiftmount)
        index = (index+8).clamp(0, self.lut_upper).to(dtype=torch.long, device=x.device)
        index = self.lut[index]

        x = index.int() * x.int()
        x = Rshift(x.mul(self.luto_req_multiplier.int()), self.luto_req_shiftmount)
        x = x.clamp(-self.act_range, self.act_range).to(torch.int8)
        return x



class LayerNorm_int(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        debug: bool = False
    ) -> None:
        '''inherit from nn.LayerNorm'''
        super().__init__(normalized_shape, eps, elementwise_affine)
        
        '''meta parameters'''
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 15
        self.wgt_bit = 8
        self.norm_out_bit = 8
        self.offset = 12

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 16383      # Signed, 2**(self.act_bit-1)-1
        self.wgt_range = 127        # Signed, 2**(self.wgt_bit-1)-1
        self.norm_out_range = 127   # Signed, 2**(self.norm_out_bit-1)-1

        '''Quantization Parameters initialization'''
        self.weight = nn.Parameter(torch.ones_like(self.weight, dtype=torch.int8), requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.zeros_like(self.bias, dtype=torch.int32), requires_grad=False)
        self.ln_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.ln_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.out_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.out_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
    

    def forward(self, x: Tensor) -> Tensor:
        '''LayerNorm with Quantize function
        '''
        assert x.dtype == torch.int16, "Input tensor must be int16 in integer inference mode!"

        norm_number = self.normalized_shape[0]
        # replace norm division to multiplication + right shift
        norm_case_check = [1280, 1024, 768, 512, 384]
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
            out = x
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
        if norm_number not in norm_case_check:
            print("norm size is: ", norm_number)
            print("Currently LayerNorm only support norm size in: ", norm_case_check)
            breakpoint()
        else:
            norm_mul = norm_multiplier[norm_number]
            norm_shmnt = norm_shiftmount[norm_number]

        mean = torch.sum(x, dim=-1, keepdim=True).int()
        mean = mul_by_shift_multiplier(mean, norm_mul) >> norm_shmnt # reduce by mul & rshift, avoid division
        sub_square = (x - mean).abs()**2

        max_bit_length = sub_square.abs().max(dim=-1, keepdim=True).values.log2().ceil().int() # max bit usage in dev_square_q's last dim (hidden dim)
        # OR you can say max clz(count leading zero) in dev_square_q's last dim
        renorm_shmnt = (max_bit_length - 16).clamp(min=0) # assume 23 bits of every number in dev_square_q's last dim is enough (fp32 has 23 bits mantissa)
        # this means, we find if anyone is more than 16 bits
        # if true, every number in the group Rshift by the same amount to prevent overflow in the following summation
        renorm_shmnt = renorm_shmnt & 0xFFFFFFFE # make it even number

        var = torch.sum((Rshift(sub_square, renorm_shmnt)), dim=-1, keepdim=True)
        var = mul_by_shift_multiplier(var, norm_mul) >> norm_shmnt # reduce by mul & rshift, avoid division
        invstd = Fast_Inverse_Square_Root(var, self.offset, 2) # this is in fixed point q20.12 format
        x = ((x - mean) * invstd)

        # x_norm_save = x.to('cpu').numpy()
        # np.save('x_norm_int.npy', x_norm_save)

        '''Re-Quantization'''
        x = Rshift( (x * self.ln_req_multiplier), (self.ln_req_shiftmount+(renorm_shmnt>>1)) )
        x = x.clamp(min=-self.act_range, max=self.act_range)

        # x_norm_req_save = x.to('cpu').numpy()
        # np.save('x_norm_req_int.npy', x_norm_req_save)

        if self.bias is None:
            bias = 0
        else:
            bias = self.bias

        x = (x.mul(self.weight.int()) + bias)

        '''Re-Quantize final output'''
        x = Rshift(x.mul(self.out_req_multiplier.int()), self.out_req_shiftmount.int())
        x = x.clamp(min=-self.norm_out_range, max=self.norm_out_range).to(torch.int8)

        return x



class Linear_int(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        encoder: bool = False,
        attn : bool = False,
        cross_attn : bool = False,
        mlp0: bool = False,
        mlp2: bool = False,
        asym: bool = True,
        Bout: int = 8,
        debug: bool = False
    ):
        '''inherit from nn.Linear'''
        super().__init__(in_features, out_features, bias)
        '''meta parameters'''
        self.encoder = encoder          # if this linear layer is in transformer encoder
        self.attn = attn                # if this linear layer is in attention module
        self.cross_attn = cross_attn    # if this linear layer is in cross-attention module
        self.mlp0 = mlp0                # if this linear layer is in MLP 1st layer
        self.mlp2 = mlp2                # if this linear layer is in MLP 2nd layer
        self.asym = asym                # if use asymmetric quantization for weights
        self.debug = debug              # debug flag

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8
        self.Bout    = Bout

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        if self.asym:
            self.register_buffer('wgt_zeros', torch.zeros(out_features))
        self.req_multiplier = torch.nn.Parameter(torch.ones((out_features), dtype=torch.uint8), requires_grad=False)
        self.req_shiftmount = torch.nn.Parameter(torch.zeros((out_features), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        self.weight = nn.Parameter(torch.ones((out_features, in_features), dtype=torch.int8), requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.ones((out_features), dtype=torch.int32), requires_grad=False)
    

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, encoder={self.encoder}, mlp2={self.mlp2}'


    def forward(self, x: Tensor) -> Tensor:
        "integer inference (all int computation)"
        assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"
        '''INT8xINT8 --> INT32 MatMul'''
        if x.device.type == 'cuda':
            out = int8_linear_matmul(x.contiguous(), self.weight.contiguous())
            
        else: # using cpu
            if _IS_ARM:
                out = w8a8.s8s8s32_gemm_packed(x.contiguous(), self.weight.contiguous(), self.out_features)
            else:
                out = torch.matmul(x.int(), self.weight.int().t())
        
        if self.bias is not None:
            '''bias addition'''
            out += self.bias
        
        if self.asym:
            '''Zero-point correction for asymmetric quantization'''
            out -= torch.sum(x.int(),dim=-1,keepdim=True).to(torch.int32) * self.wgt_zeros.to(torch.int32).reshape(1,1,-1)

        '''Re-Quantization'''
        out = Rshift(out.mul(self.req_multiplier.int()), self.req_shiftmount)
        
        if self.Bout <= 8: # re-quant to int8
            out = out.clamp(min=-(2**(self.Bout-1)), max=(2**(self.Bout-1)-1)).to(torch.int8)
        elif self.Bout <=16: # De-quant
            out = out.clamp(min=-(2**(self.Bout-1)), max=(2**(self.Bout-1)-1)).to(torch.int16)
        elif self.Bout <=32: # De-quant
            pass
        else:
            raise NotImplementedError("re-quant to more than INT64 is un-support!")
        
        return out
    
    
    def pack_weight(self):
        """Pack weight matrix for faster int8 GEMM
        """
        if _IS_ARM:
            self.weight = nn.Parameter(w8a8.pack_B_to_Bp(self.weight), requires_grad=False)

        

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
        debug: bool = False
    ):
        '''inherit from nn.Conv1d'''
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        '''meta parameters'''
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        self.weight = nn.Parameter(
            torch.ones((self.out_channels, self.in_channels * self.kernel_size[0]), dtype=torch.int8),
            requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.ones((self.out_channels), dtype=torch.int32), requires_grad=False)

    
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"
        out = torch.nn.functional.unfold(x.unsqueeze(-1).float(), kernel_size=(self.kernel_size[0],1),
            dilation=(self.dilation[0],1), padding=(self.padding[0],0), stride=(self.stride[0],1)).permute(0,2,1).to(torch.int8)
        
        if x.device.type == 'cuda':
            out = int8_linear_matmul(out.contiguous(), self.weight)
        else: # using cpu
            if _IS_ARM:
                out = w8a8.s8s8s32_gemm(out.contiguous(), self.weight)
            else:
                out = torch.matmul(out.int(), self.weight.int().t())
        
        if self.bias is not None:
            out += self.bias
        out = out.permute(0,2,1)
        '''Re-Quantization'''
        out = Rshift(out.mul(self.req_multiplier.int()), self.req_shiftmount)
        out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
        return out



class MultiHeadAttention_int(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        encoder: bool = False,
        cross_attn : bool = False,
        debug: bool = False,
    ):
        '''inherit from nn.Module'''
        super().__init__()
        '''meta parameters'''
        self.n_state = n_state
        self.n_head = n_head
        self.encoder = encoder
        self.cross_attention = cross_attn
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.qk_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.qk_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.qk_offset = 7
        self.rsqrt_offset = 10

        '''sub-modules initialization'''
        self.query = Linear_int(
            n_state, n_state, # Layer config
            encoder=encoder, attn = True)
        self.key = Linear_int(
            n_state, n_state, bias=False, # Layer config
            encoder=encoder, cross_attn = cross_attn, attn = True)
        self.value = Linear_int(
            n_state, n_state, # Layer config
            encoder=encoder, cross_attn = cross_attn, attn = True)
        self.out = Linear_int(
            n_state, n_state, # Layer config
            encoder=encoder, attn = True, Bout=32)
    

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


    def shift_exp(self,x,mask):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        else:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(
                x.shape[0]//mask.shape[0],
                x.shape[1]//mask.shape[1],
                x.shape[2]//mask.shape[2],
                x.shape[3]//mask.shape[3])

        offset = int(self.qk_offset)
        x[mask] = torch.iinfo(torch.int16).min
        x_max = x.max(dim=-1, keepdim=True).values + 1
        x_sub = x - x_max
        x_mul_fix = x_sub + (x_sub >> 1) - (x_sub >> 4)  # qk_sub_fix * b1.0111
        x_mul_int = x_mul_fix >> offset
        x_mul_frac = x_mul_fix - (x_mul_int << offset)
        x_exp = ((1 << offset) + (x_mul_frac >> 1)) >> (-x_mul_int)
        x_exp[mask] = 0
        return x_exp.to(torch.uint8)



    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape

        qk = None

        '''Calculate QK^T'''
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)

        qk = Batch_Head_s8s8s32_GeMM(q, k)  # int32 tensor
        
        qk = Rshift(qk.mul(self.qk_req_multiplier.int()), self.qk_req_shiftmount)
        qk = qk.clamp(min=-32767, max=32767).to(torch.int16)

        '''Calculate softmax(QK^T) * V'''
        if mask is not None:
            w = self.shift_exp(qk, mask[:n_ctx, :n_ctx])
        else:
            w = self.shift_exp(qk, mask)

        out = Batch_Head_u8s8s32_GeMM(w, v)  # int32 tensor
        sum_inv_mul = Fast_Inverse_Square_Root(w.sum(dim=-1, keepdim=True), self.rsqrt_offset, Newton_iter=2)
        out = Rshift(out * sum_inv_mul, self.rsqrt_offset)
        out = Rshift(out * sum_inv_mul, self.rsqrt_offset)
        out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)

        return out.permute(0, 2, 1, 3).flatten(start_dim=2)



class ResidualAttentionBlock_int(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        encoder: bool = False,
        cross_attention: bool = False,
        debug: bool = False
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.encoder = encoder
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_attn_bit = 15
        self.act_cross_attn_bit = 15
        self.act_mlp_bit = 15

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_attn_range = 16383         # Signed
        self.act_cross_attn_range = 16383   # Signed
        self.act_mlp_range = 16383          # Signed

        '''Quantization Parameters initialization'''
        self.attn_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.attn_res_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.cross_attn_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.cross_attn_res_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.mlp_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint8), requires_grad=False)
        self.mlp_res_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)

        '''sub-modules initialization'''
        # Attention Module
        self.attn = MultiHeadAttention_int(
            n_state, n_head, # Layer config
            encoder=encoder)
        self.attn_ln = LayerNorm_int(n_state)

        # Cross-Attention Module (if needed)
        self.cross_attn = (
            MultiHeadAttention_int(
                n_state, n_head, # Layer config
                encoder=encoder, cross_attn = cross_attention) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm_int(n_state) if cross_attention else None

        # MLP Module
        self.n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear_int(
                n_state, self.n_mlp, # Layer config
                encoder=encoder, mlp0 = True),
            GELU_int(channel_num = self.n_mlp, out_channel_num = self.n_mlp),
            Linear_int(
                self.n_mlp, n_state, # Layer config
                encoder=encoder,mlp2=True, Bout=32)
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
        module_inout = self.attn_ln(x)

        '''Compute Attention'''
        module_inout = self.attn(module_inout, mask=mask, kv_cache=kv_cache)

        '''Residual connection with/without quantization'''
        x = x.int().mul(self.attn_res_req_multiplier)
        x = Rshift(module_inout + x, self.attn_res_req_shiftmount)
        x = x.clamp(-self.act_attn_range, self.act_attn_range).to(torch.int16)

        '''if exists Cross-Attention Module, compute Cross-Attention'''
        if self.cross_attn:

            '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
            module_inout = self.cross_attn_ln(x)

            '''Compute Cross-Attention, because cross-attention so need xa input (xa stands for x audio)'''
            module_inout = self.cross_attn(module_inout, xa, kv_cache=kv_cache)

            '''Residual connection with/without quantization'''
            x = x.int().mul(self.cross_attn_res_req_multiplier)
            x = Rshift(module_inout + x, self.cross_attn_res_req_shiftmount)
            x = x.clamp(-self.act_cross_attn_range, self.act_cross_attn_range).to(torch.int16)

        '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
        module_inout = self.mlp_ln(x)
        
        '''Compute MLP'''
        module_inout = self.mlp(module_inout)
        
        '''Residual connection with/without quantization'''
        x = x.int().mul(self.mlp_res_req_multiplier)
        x = Rshift(module_inout + x, self.mlp_res_req_shiftmount)
        x = x.clamp(-self.act_mlp_range, self.act_mlp_range).to(torch.int16)


        return x



class AudioEncoder_int(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        debug: bool = False
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_pe_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_pe_range = 127    # Signed, 2**(self.act_pe_bit-1)-1

        '''Quantization Parameters initialization'''
        self.req_multiplier = nn.Parameter(torch.ones((n_state), dtype=torch.uint8), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros((n_state), dtype=torch.int8), requires_grad=False)

        '''sub-modules initialization'''
        self.register_buffer("positional_embedding", torch.empty((n_ctx, n_state),dtype=torch.int8))
        self.input_alpha = nn.Parameter( torch.tensor(1.0) )
        self.gelu1 = GELU_int()
        self.gelu2 = GELU_int(out_channel_num=self.n_state)
        self.conv1 = Conv1d_int(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d_int(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.blocks: Iterable[ResidualAttentionBlock_int] = nn.ModuleList(
            [ResidualAttentionBlock_int(n_state, n_head, encoder=True) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm_int(n_state)
    
    
    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        if x.dtype != torch.int8:
            # quantize input mel spectrogram to int8
            x = (x * self.input_alpha).round().clamp(min=-128, max=127).to(torch.int8)

        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.gelu2(x)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
       
        # x is already quantized in conv layers
        # add positional embedding ( offline quantized )
        x = (x.int() + self.positional_embedding)
        x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount)
        x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).to(torch.int16)
        
        '''Pass through Transformer Blocks'''
        for _ in range(self.n_layer):
            x = self.blocks[_](x)

        '''Compute Final LayerNorm'''
        x = self.ln_post(x)

        return x



class TextDecoder_int(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        debug: bool = False
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8     # positional encoding

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        self.req_multiplier = nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.clf_req_multiplier = torch.nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
        self.clf_req_shiftmount = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)

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
        x = ( self.token_embedding(x).to(torch.int32) + self.positional_embedding[offset : offset + x.shape[-1]] )
        x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount)
        x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).to(torch.int16)

        for _ in range(self.n_layer):
            x = self.blocks[_](x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)

        # int8 matmul with token embedding weight
        if torch.cuda.is_available(): # if GPU is available, use faster int8 GEMM
            x = int8_linear_matmul(x, self.token_embedding.weight)
        else:
            x = w8a8.s8s8s32_gemm(x.contiguous(), self.token_embedding.weight)
        # requant logits
        x = Rshift(x.mul(self.clf_req_multiplier.int()), self.clf_req_shiftmount)
        x= x.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
        return x



QUANT_MODULES = [
    Conv1d_int,
    Linear_int,
    GELU_int,
    LayerNorm_int,
    AudioEncoder_int,
    TextDecoder_int,
    ResidualAttentionBlock_int,
    MultiHeadAttention_int
]



class Whisper_int(nn.Module):
    def __init__(self, dims: ModelDimensions, debug: bool = False):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.dims = dims
        self.debug = debug

        '''sub-modules initialization'''
        self.encoder = AudioEncoder_int(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            self.debug
        )
        self.decoder = TextDecoder_int(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            self.debug
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
    
    def pack_linear_weights(self):
        """Pack all Linear_int weights for faster int8 GEMM
        """
        def pack_if_linear_int(module: nn.Module):
            if isinstance(module, Linear_int):
                module.pack_weight()

        self.apply(pack_if_linear_int)

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



def Build_from_Quant(Orig:Whisper_quantized)->Whisper_int:
    '''Build integer inference model from quantized model'''
    assert Orig.quantize, "Input model is not quantized model!"
    result = Orig.convert_to_int_mode()
    assert result, "Convert to integer inference model failed!"
    Model_int = Whisper_int(Orig.dims)

    result = Model_int.load_state_dict(Orig.state_dict(), strict=False)
    print("Missing keys not in CKPT: ", len(result.missing_keys), ", Showing 10: ", result.missing_keys[:10])
    print("Unexpected keys in CKPT: ", len(result.unexpected_keys), ", Showing 10: ", result.unexpected_keys[:10])

    return Model_int

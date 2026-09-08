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
import math
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from bitsandbytes.functional import int8_linear_matmul

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .kernels import w8a8
from .modules.functional import (
    Rshift,
    Fast_Inverse_Square_Root,
    Batch_Head_s8s8s32_GeMM,
    Batch_Head_u8s8s32_GeMM,
    int_bit_length,
    build_rsqrt_lut,
    LUT_Based_Inverse_Square_Root
)
from .modules.linear import Linear8, Linear32
from .model_int import LayerNorm_int

import matplotlib.pyplot as plt
import os
from typing import Any, Dict, List, Optional, Tuple, Union


'''Only Import quantized modules'''
__all__ = [
    "GELU_quantized",
    "LayerNorm_quantized",
    "Linear_quantized",
    "Conv1d_quantized",
    "MultiHeadAttention_quantized",
    "ResidualAttentionBlock_quantized",
    "AudioEncoder_quantized",
    "TextDecoder_quantized",
    "Whisper_quantized"
    ]

INT_MAX = {
    "U8": (0,torch.iinfo(torch.uint8).max),
    "S8": (-torch.iinfo(torch.int8).max, torch.iinfo(torch.int8).max),
    "U16": (0, torch.iinfo(torch.uint16).max),
    "S16": (-torch.iinfo(torch.int16).max, torch.iinfo(torch.int16).max),
    "U32": (0, torch.iinfo(torch.uint32).max),
    "S32": (-torch.iinfo(torch.int32).max, torch.iinfo(torch.int32).max),
    "U64": (0, torch.iinfo(torch.uint64).max),
    "S64": (-torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).max),
}




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


class GELU_quantized(nn.Module):
    def __init__(
        self,
        approximate: str = 'none',
        channel_num : int = 1,
        out_channel_num : int = 1,
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
        debug: bool = False,
    ) -> None:
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.approximate = approximate
        self.channel_num = channel_num
        self.calibrate = calibrate
        self.quantize = quantize  
        self.int_mode = int_mode
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8
        self.lut_index_bit = 4
        self.lut_data_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 2**(self.act_bit-1)-1
        self.lut_high = 6
        self.lut_low = -6

        '''Quantization Parameters initialization'''
        self.act_alpha = torch.nn.Parameter(torch.ones(self.channel_num))
        self.lut_delta = (self.lut_high - self.lut_low) / (2**self.lut_index_bit)
        self.lut = nn.Parameter(F.sigmoid(torch.arange(self.lut_low, self.lut_high, self.lut_delta)).mul(2**self.lut_data_bit -1).round().to(torch.uint8), requires_grad=False)
        self.lut[0] = 0
        self.lut[-1] = 2**self.lut_data_bit -1
        self.luti_req_multiplier = nn.Parameter(torch.ones(channel_num, dtype=torch.uint32), requires_grad=False)
        self.luti_req_shiftmount = nn.Parameter(torch.zeros(channel_num, dtype=torch.int8), requires_grad=False)
        self.luto_req_multiplier = nn.Parameter(torch.ones(channel_num, dtype=torch.uint8), requires_grad=False)
        self.luto_req_shiftmount = nn.Parameter(torch.zeros(channel_num, dtype=torch.int8), requires_grad=False)
        if self.quantize and self.int_mode:
            self.de_smooth_vec = nn.Parameter(torch.ones(self.channel_num, dtype=torch.uint8), requires_grad=False)
        else:
            self.de_smooth_vec = nn.Parameter(torch.ones(self.channel_num)*1.702, requires_grad=False)
    
    
    def copy_quant_params(self, gelu_quantized:'GELU_quantized')->None:
        """Copy quantization parameters from another GELU_quantized layer
        Args:
            gelu_quantized (GELU_quantized): source GELU_quantized layer
        """
        self.act_alpha.data = gelu_quantized.act_alpha.data.clone()
        self.lut.data = gelu_quantized.lut.data.clone()
        self.de_smooth_vec.data = gelu_quantized.de_smooth_vec.data.clone()


    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.running_max_input = torch.zeros(self.channel_num, device=self.lut.device, dtype=torch.float32)
        self.running_min_input = torch.zeros(self.channel_num, device=self.lut.device, dtype=torch.float32)
        self.calibrate = True

        return self.calibrate
    
    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    @torch.no_grad()
    def convert_to_int_mode(self, out_alpha)->bool:
        assert self.quantize, "Only support convert to int mode after quantization!"
        
        # caluclate requantization parameters
        luti_req_alpha = self.de_smooth_vec / self.act_alpha / self.lut_delta
        luti_req_multiplier, luti_req_shiftmount = torch.frexp(luti_req_alpha)
        self.luti_req_multiplier.data = (luti_req_multiplier * 2.**31).round().to(torch.uint32)
        self.luti_req_shiftmount.data = (-luti_req_shiftmount - 1).round().clamp(0,32).to(torch.int8)

        luto_req_alpha = out_alpha / self.act_alpha / 255.0
        #                out_act_alpha / (act_alpha * lut_out_alpha)
        luto_req_multiplier, luto_req_shiftmount = torch.frexp(luto_req_alpha)
        self.luto_req_multiplier.data = (luto_req_multiplier * 2.**7).round().to(torch.uint8)
        self.luto_req_shiftmount.data = (-luto_req_shiftmount+7).round().clamp(0,32).to(torch.int8)
        
        self.out_alpha = out_alpha
        
        self.int_mode = True
    

    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        self.quantize = True
        return self.quantize
            

    def forward(self, x: Tensor) -> Tensor:
        if self.calibrate:
            with torch.no_grad():
                if self.channel_num == 1: # GeLU in Convolution Encoder, Per-Tensor Quantization
                    input_max = x.max()
                    input_min = x.min()
                    if self.running_num==0:
                        self.running_max_input.add_(input_max)
                        self.running_min_input.add_(input_min) #for asymmetric
                        self.running_num += 1
                    else:
                        self.running_max_input.mul_(0.9).add_(input_max*0.1) 
                        self.running_min_input.mul_(0.9).add_(input_min*0.1)

                    abs_max = torch.maximum(self.running_max_input.data, self.running_min_input.data)
                    self.act_alpha.data=self.act_range/abs_max
                else: # GeLU in Transformer MLP, Per-Channel Quantization
                    input_max = x.abs().amax(dim=(0,1)).clamp_min(1e-12)
                    input_min = x.abs().amin(dim=(0,1)).clamp_min(1e-12)
                    if self.running_num==0:
                        self.running_max_input.add_(input_max)
                        self.running_min_input.add_(input_min) #for asymmetric
                        self.running_num += 1
                    else:
                        self.running_max_input.data = torch.maximum(input_max, self.running_max_input)
                        self.running_min_input.data = torch.minimum(input_min, self.running_min_input)

                    abs_max = torch.maximum(self.running_max_input.data, self.running_min_input.data)
                    self.act_alpha.data=self.act_range/abs_max

        if self.quantize:
            if self.int_mode:
                assert x.dtype == torch.int8, "x tensor must be int8 in integer inference mode!"
                # Integer Inference Mode

                '''De-Smoothing and LUT lookup (with Quantize)'''
                index = (x.long().mul(self.luti_req_multiplier.long())>>32).int()
                index = Rshift(index, self.luti_req_shiftmount)
                index = (index+(2**(self.lut_index_bit-1))).clamp(0, 2**self.lut_index_bit-1).to(dtype=torch.long, device=x.device)
                index = self.lut[index]

                x = index.int() * x.int()
                x = Rshift(x.mul(self.luto_req_multiplier.int()), self.luto_req_shiftmount)
                x = x.clamp(-self.act_range, self.act_range).to(torch.int8)
                return x
            else:
                # Quantize x
                input_t = x * self.act_alpha
                input_c = input_t.clamp(min=-127, max=127)
                input_q = torch.round(input_c).to(torch.int8)
                input_deq = input_q / self.act_alpha

                # De-Smoothing and LUT lookup (with Quantize)
                "1.702x (with de-smoothing)"
                self.sig_input = input_deq * self.de_smooth_vec
                output = F.sigmoid(self.sig_input)
                output = output.mul(2**self.lut_data_bit-1).round().int()


                "GeLU = x * sigmoid(1.702x)"
                x = input_q.int() * output.int()
                x = x.float().div(2**self.lut_data_bit-1).div(self.act_alpha)

                return x
        else:
            return x * F.sigmoid(x * self.de_smooth_vec.to(x.dtype))

    def extra_repr(self) -> str:
        return 'approximate={}'.format(repr(self.approximate))



class LayerNorm_quantized(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
        debug: bool = False
    ) -> None:
        '''inherit from nn.LayerNorm'''
        super().__init__(normalized_shape, eps, elementwise_affine)
        
        '''meta parameters'''
        self.calibrate = calibrate
        self.quantize = quantize
        self.int_mode = int_mode
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 15
        self.wgt_bit = 8
        self.bias_bit = self.act_bit + self.wgt_bit + 8
        self.norm_bit = 15
        self.out_bit = 8
        self.offset = 12

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 2**(self.act_bit-1)-1              # Signed
        self.wgt_range = 2**(self.wgt_bit-1)-1              # Signed
        self.bias_range = 2**(self.bias_bit-1)-1            # Signed
        self.norm_range = 2**(self.norm_bit-1)-1            # Signed
        self.out_range = 2**(self.out_bit-1)-1              # Signed

        '''Quantization Parameters initialization'''
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0)) # Per-Tensor Scale for input activation
        self.wgt_alpha = torch.nn.Parameter(torch.tensor(1.0))  # Per-Tensor Scale for weight (gamma)
        self.ln_alpha = torch.nn.Parameter(torch.tensor(1.0))   # Per-Tensor Scale for norm activation
        if self.quantize:
            self.weight = nn.Parameter(torch.ones_like(self.weight, dtype=torch.int8), requires_grad=False)
            if self.bias is not None:
                self.bias = nn.Parameter(torch.zeros_like(self.bias, dtype=torch.int32), requires_grad=False)
        self.ln_req_multiplier = nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
        self.ln_req_shiftmount = nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.out_req_multiplier = nn.Parameter(torch.ones(1, dtype=torch.uint32), requires_grad=False)
        self.out_req_shiftmount = nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)


    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.running_max = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
        self.running_max_ln = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
        self.calibrate = True

        return self.calibrate
    
    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    def convert_to_int_mode(self, out_alpha: torch.Tensor)->bool:
        assert self.quantize, "Only support convert to int mode after quantization!"
        
        out_req_alpha = out_alpha / (self.ln_alpha * self.wgt_alpha)
        out_req_multipler, out_req_shiftmount = torch.frexp(out_req_alpha)
        self.out_req_multiplier.data = out_req_multipler.mul(2**31).round().to(torch.uint32).view(self.out_req_multiplier.shape)
        self.out_req_shiftmount.data = (-out_req_shiftmount - 1).clamp(0,32).to(torch.int8).view(self.out_req_multiplier.shape)
        self.dequant_alpha = self.act_alpha
        
        self.int_mode = True
    

    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        ln_req_multiplier, ln_req_shiftmount = torch.frexp(self.ln_alpha)
        self.ln_req_multiplier = nn.Parameter(ln_req_multiplier.mul(2**7).round().clamp(0,255).to(torch.uint8), requires_grad=False)
        self.ln_req_shiftmount = nn.Parameter((-ln_req_shiftmount + 7 + self.offset).clamp(0,31).to(torch.int8), requires_grad=False)

        self.wgt_alpha.data = self.wgt_range/self.weight.data.abs().max()
        weight_q = self.weight.mul(self.wgt_alpha).clamp(min=-self.wgt_range, max=self.wgt_range).round().to(torch.int8)
        self.weight = nn.Parameter(weight_q, requires_grad=False)


        if self.bias is not None:
            bias_q = self.bias.mul(self.ln_alpha).mul(self.wgt_alpha).clamp(min=-self.bias_range, max=self.bias_range).round().to(torch.int32)
            self.bias = nn.Parameter(bias_q, requires_grad=False)
        del weight_q, bias_q

        self.quantize = True
        return self.quantize
        

    def forward(self, x: Tensor) -> Tensor:
        '''LayerNorm with Quantize function
        '''
        if self.calibrate:
            input_curmax = x.abs().max()
            mean = torch.mean(x.float(), dim = [-1], keepdim=True)
            var = torch.mean((x.float() - mean) ** 2, dim = [-1], keepdim=True) + self.eps
            x_norm = (x.float() - mean) * var.rsqrt()
            out_ln = x_norm.abs().max()

            with torch.no_grad():
                if self.running_num==0:
                    self.running_max.add_(input_curmax)
                    self.running_max_ln.add_(out_ln)
                    self.running_num += 1
                else:
                    self.running_max = torch.maximum(self.running_max, input_curmax)
                    self.running_max_ln = torch.maximum(self.running_max_ln, out_ln)

            self.act_alpha.data = self.act_range/self.running_max.to(x.device)
            self.ln_alpha.data=self.norm_range/self.running_max_ln.data

        if self.quantize:
            if self.int_mode:
                x_q = x.to(torch.int32)
            else:
                x_t = x * self.act_alpha
                x_c = x_t.clamp(min=-self.act_range, max=self.act_range)
                x_q = torch.round(x_c).to(torch.int32)
                x_deq = x_q.to(x.dtype)/self.act_alpha
            
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
            if norm_number not in norm_case_check:
                print("norm size is: ", norm_number)
                print("Currently LayerNorm only support norm size in: ", norm_case_check)
                breakpoint()
            else:
                norm_mul = norm_multiplier[norm_number]
                norm_shmnt = norm_shiftmount[norm_number]
            

            mean_q = torch.sum(x_q, dim=-1, keepdim=True).int()
            mean_q = mul_by_shift_multiplier(mean_q, norm_mul) >> norm_shmnt # reduce by mul & rshift, avoid division
            dev_square_q = (x_q - mean_q).abs()**2

            max_bit_length = int_bit_length(dev_square_q.max(dim=-1, keepdim=True).values) # max bit usage in dev_square_q's last dim (hidden dim)
            # OR you can say max clz(count leading zero) in dev_square_q's last dim
            renorm_shmnt = (max_bit_length - 16).clamp(min=0) # assume 23 bits of every number in dev_square_q's last dim is enough (fp32 has 23 bits mantissa)
            # this means, we find if anyone is more than 16 bits
            # if true, every number in the group Rshift by the same amount to prevent overflow in the following summation
            # renorm_shmnt = renorm_shmnt & 0xFFFFFFFE # make it even number
            renorm_shmnt = (renorm_shmnt + 1) & ~1

            var_q = torch.sum((Rshift(dev_square_q, renorm_shmnt)), dim=-1, keepdim=True).int()
            var_q = (mul_by_shift_multiplier(var_q, norm_mul) >> norm_shmnt).int() # reduce by mul & rshift, avoid division
            invstd_q = Fast_Inverse_Square_Root(var_q, self.offset, 2) # this is in fixed point q12.20 format
            # invstd_q = LUT_Based_Inverse_Square_Root(var_q, self.rsqrt_lut, self.offset, self.lut_bits, Newton_iter=2) # this is in fixed point q12.20 format
            x_norm = ((x_q - mean_q) * invstd_q)

            # x_norm_save = x_norm.to('cpu').numpy()
            # np.save('x_norm_fq.npy', x_norm_save)
            '''Re-Quantization'''
            x_norm = Rshift( (x_norm * self.ln_req_multiplier), (self.ln_req_shiftmount+(renorm_shmnt>>1)) )
            x_norm = x_norm.clamp(min=-self.norm_range, max=self.norm_range)

            if self.bias is None:
                bias = 0
            else:
                bias = self.bias
            
            out = (x_norm.mul(self.weight.int()) + bias)

            '''Re-Quantize final output'''
            if self.int_mode:
                out = (out.long().mul(self.out_req_multiplier.long())>>32).int()
                out = Rshift(out, self.out_req_shiftmount)
                out = out.clamp(min=-self.out_range, max=self.out_range).to(torch.int8)

                return out, x
            else:
                return out.float() / self.ln_alpha / self.wgt_alpha, x_deq
        else:
            return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps).type(x.dtype), x



class Linear_quantized(nn.Linear):
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
        calibrate: bool = False,
        calibrate2: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
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
        self.calibrate = calibrate      # calibrate quantization parameters flag
        self.calibrate2 = calibrate2    # observe channel max for activation smoothing flag
        self.quantize = quantize        # quantize flag
        self.int_mode = int_mode        # integer inference mode flag
        self.debug = debug              # debug flag

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.wgt_bit = 8
        self.act_bit = 8
        self.bias_bit = self.wgt_bit + self.act_bit + 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 2 ** (self.act_bit - 1) - 1        # Signed
        self.wgt_range = 2 ** (self.wgt_bit - 1) - 1        # Signed
        self.bias_range = 2 ** (self.bias_bit - 1) - 1      # Signed

        '''Quantization Parameters initialization'''
        self.register_buffer('act_alpha', torch.tensor(1.0))
        if self.asym:
            self.register_buffer('wgt_alpha', torch.ones(out_features,1))
            self.register_buffer('wgt_zeros', torch.zeros(out_features, dtype=torch.int8))
        else:
            self.register_buffer('wgt_alpha', torch.ones(out_features,1))
        self.req_multiplier = torch.nn.Parameter(torch.ones((out_features), dtype=torch.int32), requires_grad=False)
        self.req_shiftmount = torch.nn.Parameter(torch.zeros((out_features), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        if self.quantize:
            self.weight = nn.Parameter(torch.ones((out_features, in_features), dtype=torch.int8), requires_grad=False)
            if self.bias is not None:
                self.bias = nn.Parameter(torch.ones((out_features), dtype=torch.int32), requires_grad=False)
            
    
    def observe_channel_max(self)->bool:
        """Open observe channel max function for activation smoothing
        Returns:
            bool: return True if open observe channel max function for activation smoothing successfully
        """
        assert not self.quantize, "Only support observe channel max function during float model mode!"
        assert not self.calibrate, "Only support observe channel max function when not calibrating quantization parameters!"

        open_check = (self.mlp0 or self.attn or self.mlp2) # only support for MLP and Attention layers
        if open_check:
            self.running_max_c = torch.zeros(self.in_features, device=self.weight.device, dtype=self.weight.dtype) # per-channel running max initialization
            self.calibrate2 = True
        return self.calibrate2
    
    
    def stop_observe_channel_max(self)->bool:
        """Close observe channel max function for activation smoothing
        note that: this stop observing, but keep the observed values
        Returns:
            bool: return True if close observe channel max function for activation smoothing successfully
        """
        self.calibrate2 = False
        return (not self.calibrate2)
    

    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"
        assert not self.calibrate2, "Only support observe quantization parameters when not observing smooth factor!"

        # init observation buffers
        self.running_num = 0
        self.running_max = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
        self.running_min = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)

        self.calibrate = True
        return self.calibrate
    
    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    @torch.no_grad()
    def convert_to_int_mode(self, out_alpha:torch.Tensor, dequant:bool=False, is_outlinear:bool=False)->bool:
        """Convert quantized layer to integer inference mode
        Args:
            out_alpha (torch.Tensor): output activation scale
        Returns:
            bool: return True if convert to integer inference mode successfully
        """
        assert self.quantize, "Only support convert to integer inference mode after quantizing!"
        assert not self.int_mode, "Layer is already in integer inference mode!"

        req_alpha = out_alpha.double() / self.wgt_alpha.squeeze().double() / self.act_alpha.double()
        req_multiplier, req_shiftmount = torch.frexp(req_alpha)
        if is_outlinear:
            self.req_multiplier.data = req_multiplier.mul(2**31).trunc().to(torch.uint32)
            self.req_shiftmount.data = req_shiftmount.clamp(0,32).to(torch.int8)
        else:
            self.req_multiplier.data = req_multiplier.mul(2**31).trunc().to(torch.uint32)
            self.req_shiftmount.data = (-req_shiftmount - 1).clamp(0,32).to(torch.int8)
        self.out_alpha = out_alpha
        self.dequant = dequant

        self.int_mode = True
        return self.int_mode
    

    @torch.inference_mode()
    def quantize_layer(self, name: Optional[str]=None)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        if name is not None:
            print(f'Quantizing: {name}')
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        '''calculate act_alpha'''
        self.act_alpha = (self.act_range / torch.maximum(self.running_max.abs(), self.running_min.abs()))
        self.running_max = None
        self.running_min = None

        "Weight per-channel asymmetric quantization (Part of search alpha/scale)"
        if self.asym:
            # find best wgt_alpha by minimizing quantization error of each channel
            WgtMax = self.weight.max(dim=-1, keepdim=True).values
            WgtMin = self.weight.min(dim=-1, keepdim=True).values

            def calc_quant_loss(channel_max, channel_min, tensor_to_quantize):
                alpha = (self.wgt_range*2.0 / (channel_max - channel_min))
                zeros = ((-channel_max-channel_min)/2.0 * self.wgt_alpha).floor()
                # Because wgt_zeros will generate some bias when round/ceil/floor is applied
                # we appply one more iteration to adjust wgt_alpha with biased weight
                # this reduce more quantization error
                CurMax = channel_max + zeros/alpha
                CurMin = channel_min - zeros/alpha
                alpha = (self.wgt_range / torch.maximum(CurMax.abs(), CurMin.abs()))

                tensor_fakequant = tensor_to_quantize.mul(alpha).round().clamp(min=-self.wgt_range, max=self.wgt_range).div(alpha)
                return (tensor_fakequant - tensor_to_quantize).abs().abs().sum(dim=-1)

            best_wgt_max = WgtMax.clone()
            best_wgt_min = WgtMin.clone()
            best_loss = calc_quant_loss(best_wgt_max, best_wgt_min, self.weight)
            r = 1e-4
            for i in range(10):
                for j in range(10):
                    cur_wgt_max = WgtMax * ((1 - r)**i)
                    cur_wgt_min = WgtMin * ((1 - r)**j)
                    cur_loss = calc_quant_loss(cur_wgt_max, cur_wgt_min, self.weight)
                    lower_loss_pos = cur_loss <= best_loss
                    best_loss[lower_loss_pos] = cur_loss[lower_loss_pos]
                    best_wgt_max[lower_loss_pos] = cur_wgt_max[lower_loss_pos]
                    best_wgt_min[lower_loss_pos] = cur_wgt_min[lower_loss_pos]

            self.wgt_alpha.data = (self.wgt_range*2.0 / (best_wgt_max - best_wgt_min))
            self.wgt_zeros.data = ((-best_wgt_max-best_wgt_min)/2.0 * self.wgt_alpha).squeeze().trunc().to(torch.int8)
            # Because wgt_zeros will generate some bias when round/ceil/floor is applied
            # we appply one more iteration to adjust wgt_alpha with biased weight
            # this reduce more quantization error
            # best_wgt_max = best_wgt_max + self.wgt_zeros.unsqueeze(-1)/self.wgt_alpha
            # best_wgt_min = best_wgt_min - self.wgt_zeros.unsqueeze(-1)/self.wgt_alpha
            # self.wgt_alpha.data = (self.wgt_range / torch.maximum(best_wgt_max.abs(), best_wgt_min.abs()))

            cur_loss = None
            cur_wgt_max = None
            cur_wgt_min = None
            best_loss = None
            best_wgt_max = None 
            best_wgt_min = None 
            self.running_max = None
            self.running_min = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache() 
        else:
            WgtMax = self.weight.abs().max(dim=-1, keepdim=True).values
            self.wgt_alpha.data = (self.wgt_range / WgtMax)
            WgtMax = None

        '''Weight quantization (quantize weight to int8 with above alpha/scale)'''
        if self.asym:
            self.weight = nn.Parameter(
                self.weight.mul(self.wgt_alpha).add(self.wgt_zeros.unsqueeze(-1)).round().clamp(min=-self.wgt_range, max=self.wgt_range).to(torch.int8).contiguous(), 
                requires_grad=False)
        else:
            self.weight = nn.Parameter(
                self.weight.mul(self.wgt_alpha).round().clamp(min=-self.wgt_range, max=self.wgt_range).to(torch.int8).contiguous(), 
                requires_grad=False)
        gc.collect()
        torch.cuda.empty_cache()

        '''bias quant, alpha of bias is wgt_alpha * act_alpha'''
        if self.bias is not None:
            bias_t = self.bias * self.wgt_alpha.squeeze() * self.act_alpha
            bias_t = bias_t.clamp(min=-self.bias_range, max=self.bias_range)
            bias_t = torch.round(bias_t).to(torch.int32).contiguous()
            self.bias = nn.Parameter(bias_t, requires_grad=False)
            bias_t = None

        self.quantize = True
        return self.quantize


    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, encoder={self.encoder}, mlp2={self.mlp2}'


    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        "calibrate2 observe & record activation ranges for smooth"
        if self.calibrate2:
            #for smooth
            with torch.no_grad():
                cur_max_c = x.abs().amax(dim=(0, 1))
                self.running_max_c = torch.maximum(self.running_max_c, cur_max_c)
                
        "calibrate2 observe & record activation ranges, then calculate act_alpha"
        if self.calibrate:
            '''find activation ranges (min & max) in this run'''
            cur_max = x.max()
            cur_min = x.min()

            '''update running max/min'''
            if self.running_num == 0:
                self.running_max = cur_max
                self.running_min = cur_min
                self.running_num += 1
            else:
                self.running_max = torch.maximum(self.running_max, cur_max)
                self.running_min = torch.minimum(self.running_min, cur_min)
            


        "quantized inference (float scale, true int8 computation)"
        if self.quantize:
            if self.int_mode:
                assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"
            else:
                if x.dtype != torch.int8:
                    x_t = x * self.act_alpha
                    x_c = x_t.clamp(min=-self.act_range, max=self.act_range)
                    x = torch.round(x_c).to(torch.int8)
                
                
            '''INT8xINT8 --> INT32 MatMul'''
            if x.device.type == 'cuda':
                out = int8_linear_matmul(x.contiguous(), self.weight)

            else: # using cpu
                if _IS_ARM:
                    out = w8a8.s8s8s32_gemm(x.contiguous(), self.weight)
                else:
                    out = torch.matmul(x.int(), self.weight.int().t())
            
            if self.bias is not None:
                '''bias addition'''
                out += self.bias
            
            if self.asym:
                '''Zero-point correction for asymmetric quantization'''
                out -= torch.sum(x.int(),dim=-1,keepdim=True).to(torch.int32) * self.wgt_zeros.to(torch.int32).reshape(1,1,-1)
            
            if self.int_mode:
                '''Re-Quantization or De-Quantization'''
                if self.dequant:
                    out = out.float().div(self.wgt_alpha.view(1,1,-1)).div(self.act_alpha)
                else:
                    out = out.float().div(self.wgt_alpha.view(1,1,-1)).div(self.act_alpha).mul(self.out_alpha).round()
                    out = out.clamp(*INT_MAX["S8"]).to(torch.int8)
                    # out = Rshift(out.long().mul(self.req_multiplier.long()), self.req_shiftmount.long())
                    # out = out.clamp(*INT_MAX["S8"]).to(torch.int8)
            else:
                '''de-quantization'''
                out = out.float().div(self.wgt_alpha.view(1,1,-1)).div(self.act_alpha)
                
            return out
        else:
            '''FP inference'''
            out = F.linear(
                x,
                self.weight.to(x.dtype),
                self.bias.to(x.dtype) if self.bias is not None else None
            )
            return out
        


class Conv1d_quantized(nn.Conv1d):
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
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
        debug: bool = False
    ):
        '''inherit from nn.Conv1d'''
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        '''meta parameters'''
        self.calibrate = calibrate
        self.quantize = quantize
        self.int_mode = int_mode
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.wgt_bit = 8
        self.act_bit = 8
        self.bias_bit = self.wgt_bit + self.act_bit + 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.wgt_range=2**(self.wgt_bit-1)-1
        self.act_range=2**(self.act_bit-1)-1
        self.bias_range=2**(self.bias_bit-1)-1

        '''Quantization Parameters initialization'''
        self.wgt_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.req_multiplier = nn.Parameter(torch.ones((self.out_channels), dtype=torch.uint32), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros((self.out_channels), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        if self.quantize:
            self.weight = nn.Parameter(
                torch.ones((self.out_channels, self.in_channels * self.kernel_size[0]), dtype=torch.int8),
                requires_grad=False)
            if self.bias is not None:
                self.bias = nn.Parameter(torch.ones((self.out_channels), dtype=torch.int32), requires_grad=False)

    
    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.running_max = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
        self.running_min = torch.zeros(1, device=self.weight.device, dtype=self.weight.dtype)
        self.calibrate = True
        return self.calibrate
    
    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    @torch.no_grad()
    def convert_to_int_mode(self, out_alpha:torch.Tensor)->bool:
        assert self.quantize, "Only support convert to int mode after quantization!"
        
        out_req_alpha = out_alpha / self.wgt_alpha / self.act_alpha
        out_req_multipler, out_req_shiftmount = torch.frexp(out_req_alpha)
        self.req_multiplier.data = out_req_multipler.mul(2**31).trunc().to(torch.uint32)
        self.req_shiftmount.data = (-out_req_shiftmount - 1).clamp(0,32).to(torch.int8)
        self.out_alpha = out_alpha
        self.int_mode = True
        return self.int_mode

    
    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        '''calculate act_alpha & wgt_alpha'''
        # Calculate act_alpha
        self.act_alpha.data = self.act_range/self.running_max
        self.running_max = None

        # find best wgt_alpha by minimizing quantization error of each channel
        weight_img2col = self.weight.view(self.out_channels, -1)  # [C, K]
        wgt_max = weight_img2col.abs().max()

        def calc_quant_loss(channel_max, tensor_to_quantize):
            alpha = self.wgt_range / channel_max
            tensor_fakequant = tensor_to_quantize.mul(alpha).round().clamp(min=-self.wgt_range, max=self.wgt_range).div(alpha)
            return (tensor_fakequant - tensor_to_quantize).abs().sum()

        cur_wgt_max = wgt_max.clone()
        best_wgt_max = wgt_max.clone()
        best_loss = calc_quant_loss(best_wgt_max, weight_img2col)
        r = 1e-4
        for i in range(50):
            cur_wgt_max = cur_wgt_max * (1 - r)
            cur_loss = calc_quant_loss(cur_wgt_max, weight_img2col)
            # lower_loss_pos = cur_loss <= best_loss
            # best_loss[lower_loss_pos] = cur_loss[lower_loss_pos]
            # best_wgt_max[lower_loss_pos] = cur_wgt_max[lower_loss_pos]
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_wgt_max = cur_wgt_max
        self.wgt_alpha.data = self.wgt_range/best_wgt_max
        wgt_max = None
        cur_loss = None
        cur_wgt_max = None
        best_loss = None
        best_wgt_max = None 

        '''Weight quantization (quantize weight to int8 with above alpha/scale)'''
        weight_t = weight_img2col * self.wgt_alpha
        weight_t = weight_t.clamp(min = -self.wgt_range, max = self.wgt_range)
        weight_t = torch.round(weight_t).to(torch.int8)
        self.weight = nn.Parameter(weight_t, requires_grad=False)
        weight_t = None

        if self.bias is not None:
            bias_t = self.bias * self.wgt_alpha.squeeze() * self.act_alpha
            bias_t = bias_t.clamp(min = -self.bias_range, max = self.bias_range)
            bias_t = torch.round(bias_t).to(torch.int32)
            self.bias = nn.Parameter(bias_t, requires_grad=False)
            bias_t = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.quantize = True
        return self.quantize

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        if self.calibrate:
            input_max = x.max().abs()
            with torch.no_grad():
                if self.running_num==0:
                    self.running_max.add_(input_max)
                    self.running_num += 1
                else:
                    self.running_max = torch.maximum(self.running_max, input_max)

        if self.quantize:
            if self.int_mode:
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
                # out = out.float().div(self.wgt_alpha).div(self.act_alpha).mul(self.out_alpha)
                out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
                return out
            else:
                x_t = x * self.act_alpha
                x_c = x_t.clamp(min=-self.act_range, max=self.act_range)
                x_q = torch.round(x_c)
                
                out = torch.nn.functional.unfold(x_q.unsqueeze(-1), kernel_size=(self.kernel_size[0],1),
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
                out = out.permute(0,2,1).float() / self.wgt_alpha / self.act_alpha
                return out
        else:
            return super()._conv_forward(
                x,
                weight.to(x.dtype),
                None if bias is None else bias.to(x.dtype)
                )



def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)



class MultiHeadAttention_quantized(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        encoder: bool = False,
        cross_attn : bool = False,
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
        debug: bool = False,
    ):
        '''inherit from nn.Module'''
        super().__init__()
        '''meta parameters'''
        self.n_state = n_state
        self.n_head = n_head
        self.encoder = encoder
        self.cross_attention = cross_attn
        self.calibrate = calibrate
        self.quantize = quantize
        self.int_mode = int_mode
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 2**(self.act_bit-1)-1  # Signed

        '''Quantization Parameters initialization'''
        self.act_alpha_q = torch.nn.Parameter(torch.ones(1))
        self.act_alpha_k = torch.nn.Parameter(torch.ones(1))
        self.act_alpha_v = torch.nn.Parameter(torch.ones(1))
        self.qk_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.int32), requires_grad=False)
        self.qk_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.av_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.int32), requires_grad=False)
        self.av_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        # self.qk_offset = 7
        self.register_buffer(
            "qk_frac_bits",
            torch.tensor(7, dtype=torch.int32)
        )
        self.rsqrt_offset = 16
        self.lut_bits = 6

        '''sub-modules initialization'''
        self.query = Linear_quantized(
            n_state, n_state, # Layer config
            encoder=encoder, attn = True, quantize=self.quantize)
        self.key = Linear_quantized(
            n_state, n_state, bias=False, # Layer config
            encoder=encoder, cross_attn = cross_attn, attn = True, quantize=self.quantize)
        self.value = Linear_quantized(
            n_state, n_state, # Layer config
            encoder=encoder, cross_attn = cross_attn, attn = True, quantize=self.quantize)
        self.out = Linear_quantized(
            n_state, n_state, # Layer config
            encoder=encoder, attn = True, quantize=self.quantize)
    

    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.running_max_q = torch.zeros(1, device=self.query.weight.device, dtype=self.query.weight.dtype)
        self.running_max_k = torch.zeros(1, device=self.key.weight.device, dtype=self.key.weight.dtype)
        self.running_max_v = torch.zeros(1, device=self.value.weight.device, dtype=self.value.weight.dtype)
        self.running_max_qk = torch.zeros(1, device=self.key.weight.device, dtype=self.key.weight.dtype)

        self.calibrate = True
        return self.calibrate

    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    

    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        self.register_buffer(
            "rsqrt_lut",
            build_rsqrt_lut(
                lut_bits=self.lut_bits,
                frac_bits=self.rsqrt_offset,
            ).to(self.query.weight.device)
        )

        self.quantize = True
        return self.quantize
    
    
    def convert_to_int_mode(self, out_alpha)->bool:
        """Convert quantized layer to integer inference mode
        Returns:
            bool: return True if convert to integer inference mode successfully
        """
        assert self.quantize, "Only support convert to integer inference mode after quantizing!"

        self.query.convert_to_int_mode(self.act_alpha_q)
        self.key.convert_to_int_mode(self.act_alpha_k)
        self.value.convert_to_int_mode(self.act_alpha_v)
        self.out.convert_to_int_mode(out_alpha, is_outlinear=True)  # output de-quant to float

        scale = (self.n_state // self.n_head) ** -0.5
        qk_req_alpha = torch.tensor([2.**self.qk_frac_bits],dtype=torch.double,device=self.act_alpha_k.device) * torch.tensor([scale],dtype=torch.double,device=self.act_alpha_k.device) / self.act_alpha_q.double() / self.act_alpha_k.double()
        qk_req_multiplier, qk_req_shiftmount = torch.frexp(qk_req_alpha)
        self.qk_req_multiplier.data = qk_req_multiplier.mul(2**7).trunc().to(torch.int32).view(self.qk_req_multiplier.shape)
        self.qk_req_shiftmount.data = (-qk_req_shiftmount + 7).clamp(0,32).to(torch.int8).view(self.qk_req_shiftmount.shape)
        test_array = (-qk_req_shiftmount + 7)
        assert torch.all(test_array >= 0), "qk_req_multiplier must be positive!"
        
        
        av_req_alpha =  self.out.act_alpha / self.act_alpha_v
        av_req_multiplier, av_req_shiftmount = torch.frexp(av_req_alpha)
        self.av_req_multiplier.data = av_req_multiplier.mul(2**7).trunc().to(torch.int8).view(self.av_req_multiplier.shape)
        self.av_req_shiftmount.data = (-av_req_shiftmount + 7).clamp(0,32).to(torch.int8).view(self.av_req_shiftmount.shape)
        test_array = (-av_req_shiftmount + 7)
        assert torch.all(test_array >= 0), "av_req_multiplier must be positive!"
        self.av_req_alpha = av_req_alpha

        self.query = Linear8.copy_param(self.query)
        self.key = Linear8.copy_param(self.key)
        self.value = Linear8.copy_param(self.value)
        self.out = Linear32.copy_param(self.out)

        self.int_mode = True
        return self.int_mode


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
            if mask.dtype != torch.bool:
                mask = (mask==float('-inf'))
            
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(
                x.shape[0]//mask.shape[0],
                x.shape[1]//mask.shape[1],
                x.shape[2]//mask.shape[2],
                x.shape[3]//mask.shape[3])

        offset = self.qk_frac_bits
        if x.dtype == torch.int16:
            x[mask] = torch.iinfo(torch.int16).min
        elif x.dtype == torch.int32:
            x[mask] = torch.iinfo(torch.int32).min
        x_max = x.max(dim=-1, keepdim=True).values + 1
        x_sub = x - x_max
        x_mul_fix = x_sub + (x_sub >> 1) - (x_sub >> 4)  # qk_sub_fix * b1.0111
        x_mul_int = x_mul_fix >> offset
        x_mul_frac = x_mul_fix - (x_mul_int << offset)
        x_exp = ((1 << offset) + (x_mul_frac >> 1)) >> (-x_mul_int)
        x_exp[mask] = 0
        x_exp[x_sub>0] = 0
        if self.qk_frac_bits > 7:
            x_exp = x_exp.int() >> (self.qk_frac_bits - 7)
        return x_exp.to(torch.uint8)



    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape

        qk = None

        '''Calculate QK^T'''
        if self.quantize:
            scale = (n_state // self.n_head) ** -0.5
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            
            if self.int_mode:
                qk = Batch_Head_s8s8s32_GeMM(q, k)  # int32 tensor
                qk = Rshift(qk.mul(self.qk_req_multiplier), self.qk_req_shiftmount)
                qk = qk.clamp(*INT_MAX["S16"]).to(torch.int16)

            else:
                rescale_factor = scale / self.act_alpha_q / self.act_alpha_k * 2.**self.qk_frac_bits
                q_t = q * self.act_alpha_q
                q_c = q_t.clamp(min=-self.act_range, max=self.act_range)
                q_q = torch.round(q_c).to(torch.int8)

                k_t = k * self.act_alpha_k
                k_c = k_t.clamp(min=-self.act_range, max=self.act_range)
                k_q = torch.round(k_c).to(torch.int8)

                v_t = v * self.act_alpha_v
                v_c = v_t.clamp(min=-self.act_range, max=self.act_range)
                v = torch.round(v_c).to(torch.int8)
            
                qk = Batch_Head_s8s8s32_GeMM(q_q, k_q)  # int32 tensor
                qk = qk.float().mul(rescale_factor).round().clamp(*INT_MAX["S32"]).to(torch.int32)
        else:
            scale = (n_state // self.n_head) ** -0.5
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = q @ k * scale

            if self.calibrate:
                input_max_q = q.abs().max()
                input_max_k = k.abs().max()
                input_max_v = v.abs().max()
                input_max_qk = qk.abs().max().log2().floor()+1
                with torch.no_grad():
                    if self.running_num==0:
                        self.running_max_q.add_(input_max_q)
                        self.running_max_k.add_(input_max_k)
                        self.running_max_v.add_(input_max_v)
                        self.running_max_qk.add_(input_max_qk)
                        self.running_num += 1
                    else:
                        self.running_max_q = torch.maximum(self.running_max_q, input_max_q)
                        self.running_max_k = torch.maximum(self.running_max_k, input_max_k)
                        self.running_max_v = torch.maximum(self.running_max_v, input_max_v)
                        self.running_max_qk = torch.maximum(self.running_max_qk, input_max_qk)
                self.act_alpha_q.data = self.act_range/self.running_max_q.data
                self.act_alpha_k.data = self.act_range/self.running_max_k.data
                self.act_alpha_v.data = self.act_range/self.running_max_v.data

                
                # Number of fractional bits used for the INT16 QK-score representation.
                # For each layer, calibration estimates the number of integer magnitude
                # bits required by the maximum QK score. Since signed INT16 provides
                # 15 magnitude/fraction bits excluding the sign bit, all remaining bits
                # are allocated to the fractional part to maximize the precision of the
                # subsequent integer exponential approximation without overflowing INT16.
                # 2026.09.07 chen96013(Jun-An Chen, chen96013@gmail.com)
                # self.qk_frac_bits = int(15-self.running_max_qk) # WHY???
                self.qk_frac_bits.fill_(
                    int(15 - self.running_max_qk.item())
                )

        '''Calculate softmax(QK^T) * V'''
        if self.quantize:
            # Naive
            # if mask is not None:
            #     qk_softmax = F.softmax(qk.float().div(2.**self.qk_frac_bits)+mask[:n_ctx, :n_ctx], dim=-1)
            # else:
            #     qk_softmax = F.softmax(qk.float().div(2.**self.qk_frac_bits), dim=-1)
            
            # qk_softmax_q = qk_softmax.mul(2.**7).round().clamp(min=0, max=255).to(torch.uint8)
            # out = Batch_Head_u8s8s32_GeMM(qk_softmax_q, v)  # int32 tensor
            # out = Rshift(out, 7)

            # if self.int_mode:
            #     out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
            # else:
            #     '''dequantize'''
            #     out = out.float() / self.act_alpha_v

            # return out.permute(0, 2, 1, 3).flatten(start_dim=2)

            # I-Whisper
            if mask is not None:
                qk_exp = self.shift_exp(qk, mask[:n_ctx, :n_ctx])
            else:
                qk_exp = self.shift_exp(qk, mask)

            out = Batch_Head_u8s8s32_GeMM(qk_exp, v)  # int32 tensor
            sum_inv_mul = Fast_Inverse_Square_Root(qk_exp.sum(dim=-1, keepdim=True).int(), self.rsqrt_offset, Newton_iter=2)

            out = Rshift(out.mul(sum_inv_mul).int(), self.rsqrt_offset)
            out = Rshift(out.mul(sum_inv_mul).int(), self.rsqrt_offset)

            if self.int_mode:
                out = Rshift(out.mul(self.av_req_multiplier).int(), self.av_req_shiftmount)
                out = out.clamp(min=-self.act_range, max=self.act_range).to(torch.int8)
            else:
                '''dequantize'''
                out = out.float() / self.act_alpha_v

            return out.permute(0, 2, 1, 3).flatten(start_dim=2)
        else:
            '''Apply mask if provided'''
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)



class ResidualAttentionBlock_quantized(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        encoder: bool = False,
        cross_attention: bool = False,
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
        debug: bool = False
    ):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.encoder = encoder
        self.calibrate = calibrate
        self.quantize = quantize
        self.int_mode = int_mode
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_attn_bit = 15
        self.act_cross_attn_bit = 15
        self.act_mlp_bit = 15

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_attn_range = 2**(self.act_attn_bit-1)-1                # Signed
        self.act_cross_attn_range = 2**(self.act_cross_attn_bit-1)-1    # Signed
        self.act_mlp_range = 2**(self.act_mlp_bit-1)-1                  # Signed

        '''Quantization Parameters initialization'''
        self.act_attn_alpha = torch.nn.Parameter(torch.ones(1))
        self.act_cross_attn_alpha = torch.nn.Parameter(torch.ones(1))
        self.act_alpha_mlp = torch.nn.Parameter(torch.ones(1))
        self.attn_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint16), requires_grad=False)
        self.attn_res_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.cross_attn_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint16), requires_grad=False)
        self.cross_attn_res_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)
        self.mlp_res_req_multiplier = nn.Parameter(torch.ones((1), dtype=torch.uint16), requires_grad=False)
        self.mlp_res_req_shiftmount = nn.Parameter(torch.zeros((1), dtype=torch.int8), requires_grad=False)

        '''sub-modules initialization'''
        # Attention Module
        self.attn = MultiHeadAttention_quantized(
            n_state, n_head, # Layer config
            encoder=encoder, quantize=self.quantize)
        self.attn_ln = LayerNorm_quantized(
            n_state, # Layer config
            quantize=self.quantize)

        # Cross-Attention Module (if needed)
        self.cross_attn = (
            MultiHeadAttention_quantized(
                n_state, n_head, # Layer config
                encoder=encoder, cross_attn = cross_attention, quantize=self.quantize) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm_quantized(
            n_state, # Layer config
            quantize=self.quantize) if cross_attention else None

        # MLP Module
        self.n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear_quantized(
                n_state, self.n_mlp, # Layer config
                encoder=encoder, mlp0 = True, quantize=self.quantize),
            GELU_quantized(
                channel_num = self.n_mlp, # Layer config
                quantize=self.quantize),
            Linear_quantized(
                self.n_mlp, n_state, # Layer config
                encoder=encoder,mlp2=True, quantize=self.quantize)
        )
        self.mlp_ln = LayerNorm_quantized(
            n_state, # Layer config
            quantize=self.quantize)


    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.running_max_attn = torch.zeros(1, device=self.attn.query.weight.device, dtype=self.attn.query.weight.dtype)
        if self.cross_attn:
            self.running_max_cross_attn = torch.zeros(1, device=self.attn.query.weight.device, dtype=self.attn.query.weight.dtype)
        self.running_max_mlp_out = torch.zeros(1, device=self.attn.query.weight.device, dtype=self.attn.query.weight.dtype)

        self.calibrate = True
        return self.calibrate

    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    def convert_to_int_mode(self, out_alpha)->bool:
        """Convert quantized layer to integer inference mode
        Returns:
            bool: return True if convert to integer inference mode successfully
        """
        assert self.quantize, "Only support convert to integer inference mode after quantizing!"

        '''Convert attention module'''
        self.attn_ln.convert_to_int_mode(self.attn.query.act_alpha)
        if self.cross_attn:
            self.attn.convert_to_int_mode(self.cross_attn_ln.act_alpha*2.**16)
        else:
            self.attn.convert_to_int_mode(self.mlp_ln.act_alpha*2.**16)

        '''Convert cross attention module (if exist)'''
        if self.cross_attn:
            self.cross_attn_ln.convert_to_int_mode(self.cross_attn.query.act_alpha)
            self.cross_attn.convert_to_int_mode(self.mlp_ln.act_alpha*2.**16)

        '''Convert MLP module'''
        self.mlp_ln.convert_to_int_mode(self.mlp[0].act_alpha)
        self.mlp[0].convert_to_int_mode(self.mlp[1].act_alpha)
        self.mlp[1].convert_to_int_mode(self.mlp[2].act_alpha)
        self.mlp[2].convert_to_int_mode(out_alpha*2.**16, is_outlinear=True)  # output de-quant to float
        self.mlp[0] = Linear8.copy_param(self.mlp[0])
        self.mlp[2] = Linear32.copy_param(self.mlp[2])
        self.out_alpha = out_alpha
        gc.collect()
        torch.cuda.empty_cache()


        '''Calculate re-scale multiplier of residual path beside Attention module'''
        if self.cross_attn:
            rescale_multiplier = self.cross_attn_ln.act_alpha*2.**15 / self.attn_ln.act_alpha
            self.attn_res_req_multiplier.data = rescale_multiplier.round().clamp(*INT_MAX["U16"]).to(torch.uint16)
        else:
            rescale_multiplier = self.mlp_ln.act_alpha*2.**15 / self.attn_ln.act_alpha
            self.attn_res_req_multiplier.data = rescale_multiplier.round().clamp(*INT_MAX["U16"]).to(torch.uint16)

        '''Calculate re-scale multiplier of residual path beside Cross-Attention module (if exist)'''
        if self.cross_attn:
            cross_attn_rescale_multiplier = self.mlp_ln.act_alpha*2.**15 / self.cross_attn_ln.act_alpha
            self.cross_attn_res_req_multiplier.data = cross_attn_rescale_multiplier.round().clamp(*INT_MAX["U16"]).to(torch.uint16)

        '''Calculate re-scale multiplier of residual path beside MLP module'''
        mlp_rescale_multiplier = out_alpha*2.**15 / self.mlp_ln.act_alpha
        self.mlp_res_req_multiplier.data = mlp_rescale_multiplier.round().clamp(*INT_MAX["U16"]).to(torch.uint16)

        self.attn_ln = LayerNorm_int.copy_param(self.attn_ln)
        self.mlp_ln = LayerNorm_int.copy_param(self.mlp_ln)
        if self.cross_attn:
            self.cross_attn_ln = LayerNorm_int.copy_param(self.cross_attn_ln)
        self.int_mode = True
        return self.int_mode
    

    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        self.quantize = True
        return self.quantize


    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
        attn_ln, x = self.attn_ln(x)

        '''Compute Attention'''
        attn_t = self.attn(attn_ln, mask=mask, kv_cache=kv_cache)

        if self.calibrate:
            '''Calibrate Attention output activation quantization parameters'''
            input_attn_res_max = attn_t.abs().max()
            with torch.no_grad():
                if self.running_num==0:
                    self.running_max_attn.add_(input_attn_res_max)
                else:
                    self.running_max_attn = torch.maximum(self.running_max_attn, input_attn_res_max)
            self.act_attn_alpha.data = torch.tensor([self.act_attn_range/self.running_max_attn.data]).to(x.device)

        '''Residual connection with/without quantization'''
        if self.quantize:
            if self.int_mode:
                x = x.int().mul(self.attn_res_req_multiplier.int())
                x = Rshift(attn_t + x, 15)
                x = x.clamp(-self.act_attn_range, self.act_attn_range).to(torch.int16)
            else:
                attn_t = attn_t.mul(self.act_attn_alpha).round().div(self.act_attn_alpha)
                x = x + attn_t
            # attn_t = attn_t.mul(self.act_attn_alpha).round().div(self.act_attn_alpha)
            # x = x + attn_t
        else:
            x = x + attn_t

        '''if exists Cross-Attention Module, compute Cross-Attention'''
        if self.cross_attn:

            '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
            cross_attn_ln, x = self.cross_attn_ln(x)

            '''Compute Cross-Attention, because cross-attention so need xa input (xa stands for x audio)'''
            cross_attn_t = self.cross_attn(cross_attn_ln, xa, kv_cache=kv_cache)

            if self.calibrate:
                '''Calibrate Cross-Attention output activation quantization parameters'''
                input_cross_attn_res_max = cross_attn_t.abs().max()
                with torch.no_grad():
                    if self.running_num ==0:
                        self.running_max_cross_attn.add_(input_cross_attn_res_max)
                    else:
                        self.running_max_cross_attn = torch.maximum(self.running_max_cross_attn, input_cross_attn_res_max)
                self.act_cross_attn_alpha.data = torch.tensor([self.act_cross_attn_range/self.running_max_cross_attn.data]).to(x.device)

            '''Residual connection with/without quantization'''
            if self.quantize:
                if self.int_mode:
                    x = x.int().mul(self.cross_attn_res_req_multiplier.int())
                    x = Rshift(cross_attn_t + x, 15)
                    x = x.clamp(-self.act_cross_attn_range, self.act_cross_attn_range).to(torch.int16)
                else:
                    cross_attn_t = cross_attn_t.mul(self.act_cross_attn_alpha).round().div(self.act_cross_attn_alpha)
                    x = x + cross_attn_t
                # cross_attn_t = cross_attn_t.mul(self.act_cross_attn_alpha).round().div(self.act_cross_attn_alpha)
                # x = x + cross_attn_t
            else:
                x = x + cross_attn_t

        '''Compute LayerNorm, and LayerNorm returns both normalized output and non-quantized/quantized input'''
        mlp_out, x = self.mlp_ln(x)
        
        '''Compute MLP'''
        mlp_out = self.mlp(mlp_out)
        
        if self.calibrate:
            '''Calibrate MLP output activation quantization parameters'''
            mlp_out_curmax = mlp_out.abs().max()
            with torch.no_grad():
                if self.running_num==0:
                    self.running_max_mlp_out.add_(mlp_out_curmax)
                    self.running_num += 1
                else:
                    self.running_max_mlp_out = torch.maximum(self.running_max_mlp_out, mlp_out_curmax)
            self.act_alpha_mlp.data=self.act_mlp_range/self.running_max_mlp_out.data

        '''Residual connection with/without quantization'''
        if self.quantize:
            if self.int_mode:
                x = x.int().mul(self.mlp_res_req_multiplier.int())
                x = Rshift(mlp_out + x, 15)
                x = x.clamp(-self.act_mlp_range, self.act_mlp_range).to(torch.int16)
                # x = x + mlp_out
            else:
                mlp_out = mlp_out.mul(self.act_alpha_mlp).round().div(self.act_alpha_mlp)
                x = x + mlp_out
            # mlp_out = mlp_out.mul(self.act_alpha_mlp).round().div(self.act_alpha_mlp)
            # x = x + mlp_out
        else:
            x = x + mlp_out

        return x



class AudioEncoder_quantized(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
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
        self.calibrate = calibrate
        self.quantize = quantize
        self.int_mode = int_mode
        self.debug = debug

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.act_pe_bit = 8

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_pe_range = 2**(self.act_pe_bit-1)-1

        '''Quantization Parameters initialization'''
        self.running_num = 0
        self.input_alpha = torch.nn.Parameter(torch.ones(1))
        self.act_alpha = torch.nn.Parameter(torch.ones(n_state))
        self.req_multiplier = nn.Parameter(torch.ones((n_state), dtype=torch.uint16), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros((n_state), dtype=torch.int8), requires_grad=False)

        '''sub-modules initialization'''
        if self.quantize:
            self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state).to(torch.int8))
        else:
            self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        if self.quantize and self.int_mode:
            self.gelu1 = GELU_quantized(calibrate=self.calibrate, quantize=self.quantize, int_mode=self.int_mode)
            self.gelu2 = GELU_quantized(out_channel_num=self.n_state, calibrate=self.calibrate, quantize=self.quantize, int_mode=self.int_mode)
        else:
            self.conv_gelu = GELU_quantized(calibrate=self.calibrate, quantize=self.quantize)
        self.conv1 = Conv1d_quantized(n_mels, n_state, kernel_size=3, padding=1, calibrate=self.calibrate, quantize=self.quantize)
        self.conv2 = Conv1d_quantized(n_state, n_state, kernel_size=3, stride=2, padding=1, calibrate=self.calibrate, quantize=self.quantize)
        self.blocks: Iterable[ResidualAttentionBlock_quantized] = nn.ModuleList(
            [ResidualAttentionBlock_quantized(n_state, n_head, encoder=True, calibrate=self.calibrate, quantize=self.quantize) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm_quantized(
            n_state, # Layer config
            quantize=self.quantize)
    
    
    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.running_max_x = torch.zeros(self.n_state, device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)

        self.calibrate = True
        return self.calibrate

    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    @torch.no_grad()
    def convert_to_int_mode(self, out_alpha)->bool:
        """Convert quantized layer to integer inference mode
        Returns:
            bool: return True if convert to integer inference mode successfully
        """
        assert self.quantize, "Only support convert to integer inference mode after quantizing!"

        self.input_alpha = nn.Parameter( self.conv1.act_alpha )
        self.gelu1 = GELU_quantized(quantize=True)
        self.gelu2 = GELU_quantized(out_channel_num=self.n_state, quantize=True)
        self.gelu1.copy_quant_params(self.conv_gelu)
        self.gelu2.copy_quant_params(self.conv_gelu)
        self.conv1.convert_to_int_mode(self.conv_gelu.act_alpha)
        self.gelu1.convert_to_int_mode(self.conv2.act_alpha)
        self.conv2.convert_to_int_mode(self.conv_gelu.act_alpha)
        self.gelu2.convert_to_int_mode(self.act_alpha.squeeze())
        for i in range(len(self.blocks)):
            self.blocks[i].convert_to_int_mode(self.blocks[i+1].attn_ln.act_alpha if i+1 < len(self.blocks) else self.ln_post.act_alpha)
        self.ln_post.convert_to_int_mode(out_alpha)
        self.ln_post = LayerNorm_int.copy_param(self.ln_post)

        self.int_mode = True
        return self.int_mode
    

    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        # find best act_pe_alpha by minimizing quantization error of each channel
        pe_max = self.positional_embedding.abs().amax(dim=0)  # [C]

        def calc_quant_loss(channel_max, tensor_to_quantize):
            alpha = self.act_pe_range / channel_max
            tensor_fakequant = tensor_to_quantize.mul(alpha).round().clamp(min=-self.act_pe_range, max=self.act_pe_range).div(alpha)
            return (tensor_fakequant - tensor_to_quantize).abs().sum(dim=0)

        cur_pe_max = pe_max.clone()
        best_pe_max = pe_max.clone()
        best_loss = calc_quant_loss(best_pe_max, self.positional_embedding)
        r = 1e-5
        for i in range(50):
            cur_pe_max = cur_pe_max * (1 - r)
            cur_loss = calc_quant_loss(cur_pe_max, self.positional_embedding)
            lower_loss_pos = cur_loss <= best_loss
            best_loss[lower_loss_pos] = cur_loss[lower_loss_pos]
            best_pe_max[lower_loss_pos] = cur_pe_max[lower_loss_pos]

        act_pe_alpha = (self.act_pe_range / best_pe_max)
        self.act_alpha.data = torch.minimum(self.act_alpha, act_pe_alpha)

        self.positional_embedding = self.positional_embedding.mul(self.act_alpha).clamp(min=-self.act_pe_range, max=self.act_pe_range).to(torch.int8)
        req_alpha = (self.blocks[0].attn_ln.act_alpha / self.act_alpha)
        req_multiplier, req_shiftmount = torch.frexp(req_alpha)
        self.req_multiplier.data = torch.round(req_multiplier * 2. ** 15).to(torch.uint16)
        self.req_shiftmount.data = (-req_shiftmount + 15).clamp(0,31).to(torch.int8)

        self.quantize = True
        return self.quantize

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        if self.int_mode and self.quantize:
            if x.dtype != torch.int8:
                x = x.mul(self.input_alpha).round().clamp(*INT_MAX["S8"]).to(torch.int8)
            
            x = self.conv1(x)
            x = self.gelu1(x)
            x = self.conv2(x)
            x = x.permute(0, 2, 1)
            x = self.gelu2(x)
        else:
            x = self.conv_gelu(self.conv1(x))
            x = self.conv_gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

       
        if self.calibrate:
            '''Calibrate input x and positional embedding quantization parameters'''
            x_abs_max_c  = x.abs().amax(dim=(0, 1))

            with torch.no_grad():
                if self.running_num == 0:
                    self.running_max_x.add_(x_abs_max_c)
                    self.running_num += 1
                else:
                    r = 0.1
                    self.running_max_x.mul_(1 - r).add_(x_abs_max_c * r)

                eps = 1e-12
                self.act_alpha.data = (self.act_pe_range / self.running_max_x.clamp_min(eps))   # [C]

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
       
        if self.quantize:
            if self.int_mode:
                # x is already quantized in conv layers
                # add positional embedding ( offline quantized )
                x = (x.int() + self.positional_embedding)
                x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount.int())
                x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).to(torch.int16)
                # x = x.float().div(self.blocks[0].attn_ln.act_alpha)
            else:
                # quant
                x = x.mul(self.act_alpha).clamp(min=-self.act_pe_range, max=self.act_pe_range) # [B,T,C]
                x = torch.round(x) # [-127,127]

                # add positional embedding ( offline quantized )
                x = (x.int() + self.positional_embedding)
                x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount.int())
                x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).to(torch.int16)

                # de-quant
                x = x.float().div(self.blocks[0].attn_ln.act_alpha)
        else:
            x = (x + self.positional_embedding.to(x.dtype))
        
        '''Pass through Transformer Blocks'''
        for _ in range(self.n_layer):
            x = self.blocks[_](x)

        '''Compute Final LayerNorm'''
        x, _ = self.ln_post(x)

        return x



class TextDecoder_quantized(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        calibrate: bool = False,
        quantize: bool = False,
        int_mode: bool = False,
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
        self.calibrate = calibrate
        self.quantize = quantize
        self.int_mode = int_mode
        self.debug = debug
        self.dequant_need = True

        '''Quantization Hyper-parameters (Bit-width of Parameters)'''
        self.wgt_bit = 8        # embedding
        self.act_pe_bit = 8     # positional encoding
        self.act_cf_bit = 8     # classifier
        self.act_logits_bit = 8 # logits output

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.wgt_range = 2**(self.wgt_bit-1)-1                  # Signed
        self.act_cf_range = 2**(self.act_cf_bit-1)-1            # Signed
        self.act_logits_range = 2**(self.act_logits_bit-1)-1    # Signed

        '''Quantization Parameters initialization'''
        self.wgt_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.act_cf_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.act_logits_alpha = torch.nn.Parameter(torch.tensor(1.0))
        
        self.req_multiplier = nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
        self.req_shiftmount = nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.clf_req_multiplier = torch.nn.Parameter(torch.ones(1, dtype=torch.uint32), requires_grad=False)
        self.clf_req_shiftmount = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)

        if self.quantize:
            self.token_embedding = nn.Embedding(n_vocab, n_state)
            self.token_embedding.weight = nn.Parameter(
                self.token_embedding.weight.to(torch.int8).contiguous(),
                requires_grad=False
            )
            self.positional_embedding = nn.Parameter(torch.empty((n_ctx, n_state), dtype=torch.int8), requires_grad=False)
        else:
            self.token_embedding = nn.Embedding(n_vocab, n_state)
            self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock_quantized] = nn.ModuleList(
            [
                ResidualAttentionBlock_quantized(n_state, n_head, cross_attention=True, calibrate=self.calibrate, quantize=self.quantize)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm_quantized(
            n_state, # Layer config
            quantize=self.quantize)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
    

    def calibrate_quant_params(self)->bool:
        """Open calibrate quantization parameters function
        Returns:
            bool: return True if open calibrate quantization parameters function successfully
        """
        assert not self.quantize, "Only support calibrate quantization parameters during float model mode!"

        # init observation buffers
        self.running_num = 0
        self.calibrate = True
        self.running_max_cf = torch.zeros(1, device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
        self.running_max_logits = torch.zeros(1, device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)

        return self.calibrate
    
    
    def stop_calibrate(self)->bool:
        """Close calibrate quantization parameters function
        note that: this stop calibration, but keep the calibrated values
        Returns:
            bool: return True if stop calibrate quantization parameters function successfully
        """
        self.calibrate = False
        return (not self.calibrate)
    
    
    @torch.no_grad()
    def convert_to_int_mode(self)->bool:
        """Convert quantized layer to integer inference mode
        Returns:
            bool: return True if convert to integer inference mode successfully
        """
        assert self.quantize, "Only support convert to integer inference mode after quantizing!"

        for i in range(len(self.blocks)):
            self.blocks[i].convert_to_int_mode(self.blocks[i+1].attn_ln.act_alpha if i+1 < len(self.blocks) else self.ln.act_alpha)
        
        self.ln.convert_to_int_mode(self.act_cf_alpha)
        self.ln = LayerNorm_int.copy_param(self.ln)
        clf_req_alpha = self.act_logits_alpha / self.wgt_alpha / self.act_cf_alpha
        clf_req_alpha, clf_req_shmnt = torch.frexp(clf_req_alpha)
        clf_req_alpha = torch.round(clf_req_alpha * 2. ** 31).clamp(0,torch.iinfo(torch.uint32).max).to(torch.uint32)
        clf_req_shmnt = (-clf_req_shmnt - 1).clamp(0,31).to(torch.int8)
        self.clf_req_multiplier.data = clf_req_alpha.view(self.clf_req_multiplier.shape)
        self.clf_req_shiftmount.data = clf_req_shmnt.view(self.clf_req_shiftmount.shape)
        
        self.mask = (self.mask==float('-inf'))

        self.int_mode = True
        return self.int_mode
    

    def quantize_layer(self)->bool:
        """Open quantization function
        Returns:
            bool: return True if open quantization function successfully
        """
        # CHECK: self.running_num exist
        assert hasattr(self, 'running_num'), "Please calibrate quantization parameters before quantizing!"
        # CHECK: self.running_num > 0
        assert self.running_num > 0, "Please calibrate quantization parameters before quantizing!"

        # wgt_alpha is alpha of token embedding weight
        # find best wgt_alpha by minimizing quantization error of each channel
        wgt_max = self.token_embedding.weight.abs().max()  # [C]

        def calc_quant_loss(channel_max, tensor_to_quantize):
            alpha = self.wgt_range / channel_max
            tensor_fakequant = tensor_to_quantize.mul(alpha).round().clamp(min=-self.wgt_range, max=self.wgt_range).div(alpha)
            return (tensor_fakequant - tensor_to_quantize).abs().sum()

        cur_wgt_max = wgt_max.clone()
        best_wgt_max = wgt_max.clone()
        best_loss = calc_quant_loss(best_wgt_max, self.token_embedding.weight)
        r = 1e-4
        for i in range(30):
            cur_wgt_max = cur_wgt_max * (1 - r)
            cur_loss = calc_quant_loss(cur_wgt_max, self.token_embedding.weight)
            lower_loss_pos = cur_loss <= best_loss
            if lower_loss_pos:
                best_loss = cur_loss
                best_wgt_max = cur_wgt_max
        self.wgt_alpha.data = self.wgt_range/best_wgt_max
        wgt_max = None
        cur_loss = None
        cur_wgt_max = None
        best_loss = None
        best_wgt_max = None
        self.running_max = None

        self.token_embedding.weight = nn.Parameter(
            self.token_embedding.weight.mul(self.wgt_alpha).round().clamp(min = -self.wgt_range, max = self.wgt_range).to(torch.int8).contiguous(),
            requires_grad=False
        )
        self.positional_embedding = nn.Parameter(self.positional_embedding.mul(self.wgt_alpha).round().clamp(min = -self.wgt_range, max = self.wgt_range).to(torch.int8), requires_grad=False)

        req_alpha = (self.blocks[0].attn_ln.act_alpha / self.wgt_alpha)
        req_multiplier, req_shiftmount = torch.frexp(req_alpha)
        self.req_multiplier.data = torch.round(req_multiplier * 2. ** 7).clamp(0,255).to(torch.uint8)
        self.req_shiftmount.data = (-req_shiftmount + 7).clamp(0,31).to(torch.int8)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.quantize = True
        return self.quantize


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
        if self.quantize:
            x = ( self.token_embedding(x).to(torch.int32) + self.positional_embedding[offset : offset + x.shape[-1]] )
            x = Rshift(x.mul(self.req_multiplier.int()), self.req_shiftmount.int())
                
            if self.int_mode:
                # re-quant
                x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).to(torch.int16)
                # x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).float().div(self.blocks[0].attn_ln.act_alpha)
            else:
                # de-quant
                x = x.clamp(min=-self.blocks[0].attn_ln.act_range, max=self.blocks[0].attn_ln.act_range).float().div(self.blocks[0].attn_ln.act_alpha)
        else:
            x = ( self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]] ).to(xa.dtype)

        for _ in range(self.n_layer):
            x = self.blocks[_](x, xa, mask=self.mask, kv_cache=kv_cache)

        x, _ = self.ln(x)

        if self.calibrate:
            '''Calibrate classifier input activation quantization parameters'''
            '''Why calibrate here? Because classifier of Whisper is not individual Linear layer'''
            '''Classifier is implemented by multiplying token embeddings weight matrix'''
            input_max = x.abs().max()
            with torch.no_grad():
                if self.running_num==0:
                    self.running_max_cf.add_(input_max)
                else:
                    if input_max >self.running_max_cf:
                        self.running_max_cf.mul_(0.9).add_(input_max*0.1)
            self.act_cf_alpha.data=torch.tensor([self.act_cf_range/self.running_max_cf.data]).to(x.device)
            
        if self.quantize:
            if self.int_mode:
                # x is already quantized into int8 in self.ln

                # int8 matmul with token embedding weight
                if x.device.type == 'cuda':
                    logits = int8_linear_matmul(x, self.token_embedding.weight)
                else:
                    if _IS_ARM:
                        logits = w8a8.s8s8s32_gemm(x, self.token_embedding.weight)
                    else:
                        logits = torch.matmul(x.int(), self.token_embedding.weight.int().t())

                # requant logits
                logits = (logits.long().mul(self.clf_req_multiplier.long()) >> 32).int()
                logits = Rshift(logits, self.clf_req_shiftmount)
                if self.dequant_need:
                    logits = logits.float().div(self.act_logits_alpha)
                else:
                    logits = logits.clamp(min=-self.act_logits_range, max=self.act_logits_range).to(torch.int8)
            else:
                # quant
                act_cf_t = x * self.act_cf_alpha
                act_cf_t_c = act_cf_t.clamp(min = -self.act_cf_range, max = self.act_cf_range)
                act_cf_q = torch.round(act_cf_t_c).to(torch.int8)
                
                if x.device.type == 'cuda':
                    logits = int8_linear_matmul(act_cf_q, self.token_embedding.weight).float()
                else:
                    if _IS_ARM:
                        logits = w8a8.s8s8s32_gemm(act_cf_q, self.token_embedding.weight).float()
                    else:
                        logits = torch.matmul(act_cf_q.int(), self.token_embedding.weight.int().t()).float()
                
                logits = logits.div(self.act_cf_alpha).div(self.wgt_alpha)
                logits = logits.mul(self.act_logits_alpha.type(logits.dtype)).round().div(self.act_logits_alpha.type(logits.dtype))
        else:
                logits = ( x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1) ).float()

        if self.calibrate:
            '''Calibrate logits output activation quantization parameters'''
            input_max = torch.max(x.abs().max())
            with torch.no_grad():
                if self.running_num==0:
                    self.running_max_logits.add_(logits.abs().max())
                    self.running_num += 1
                else:
                    self.running_max_logits = torch.maximum(self.running_max_logits, logits.abs().max())
                self.act_logits_alpha.data=torch.tensor([self.act_logits_range/self.running_max_logits.data]).to(x.device)

        return logits



QUANT_MODULES = [
    Conv1d_quantized,
    Linear_quantized,
    GELU_quantized,
    LayerNorm_quantized,
    AudioEncoder_quantized,
    TextDecoder_quantized,
    ResidualAttentionBlock_quantized,
    MultiHeadAttention_quantized
]



class Whisper_quantized(nn.Module):
    def __init__(self, dims: ModelDimensions, calibrate: bool = False, quantize: bool = False, debug: bool = False):
        '''inherit from nn.Module'''
        super().__init__()

        '''meta parameters'''
        self.dims = dims
        self.calibrate = calibrate
        self.quantize = quantize
        self.debug = debug

        '''sub-modules initialization'''
        self.encoder = AudioEncoder_quantized(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            self.calibrate,
            self.quantize,
            self.debug
        )
        self.decoder = TextDecoder_quantized(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            self.calibrate,
            self.quantize,
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
            if isinstance(layer, MultiHeadAttention_quantized):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function


    def model_open_calibrate(self):
        success = []
        for name, m in self.named_modules():
            if type(m) in QUANT_MODULES:
                success.append(m.calibrate_quant_params())
        return all(success)


    def model_close_calibrate(self):
        success = []
        for m in self.modules():
            if type(m) in QUANT_MODULES: # all
                success.append(m.stop_calibrate())
        return all(success)
    
    
    def convert_to_int_mode(self):
        success = []
        success.append(self.encoder.convert_to_int_mode(self.decoder.blocks[0].cross_attn.key.act_alpha))
        success.append(self.decoder.convert_to_int_mode())
        return all(success)
        

    def quantize_model(self, logger: Logger=None)->bool:
        success = []
        for m in self.modules():
            if type(m) in QUANT_MODULES:
                success.append(m.quantize_layer())
        self.quantize = True
        return all(success)


    def model_open_quant_debug(self):
        for m in self.modules():
            if type(m) in QUANT_MODULES:
                m.debug = True


    def observe_channel_max(self):
        for m in self.modules():
            if isinstance(m, Linear_quantized):
                success = m.observe_channel_max()


    def stop_observe_channel_max(self):
        for m in self.modules():
            if isinstance(m, Linear_quantized):
                success = m.stop_observe_channel_max()
    
    def check_quantized(self, logger: Logger)->bool:
        """Check whether the model is in quantized mode
        Returns:
            bool: return True if the model is in quantized mode
        """
        check = True
        for n, m in self.named_modules():
            if hasattr(m, 'quantize'):
                if m.quantize:
                    logger.debug(f"QUANTIZED CHECK: PASS {n}")
                else:
                    logger.debug(f"QUANTIZED CHECK: FAIL {n}")
                    check = False
        return check
        
import torch
from torch import nn
from bitsandbytes.functional import int8_linear_matmul

from .functional import Rshift
from i_whisper.kernels import _C



class LinearBase(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        asym: bool = True,
        backend="torch"
    ):
        '''inherit from nn.Linear'''
        super().__init__(in_features, out_features, bias)
        '''meta parameters'''
        self.backend = backend          # backend for computation
        self.asym = asym                # if use asymmetric quantization for weights

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        if self.asym:
            self.register_buffer('wgt_zeros', torch.zeros((out_features), dtype=torch.int8))
        else:
            self.wgt_zeros = None
        self.req_multiplier = torch.nn.Parameter(torch.ones((out_features), dtype=torch.uint32), requires_grad=False)
        self.req_shiftmount = torch.nn.Parameter(torch.zeros((out_features), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        self.weight = nn.Parameter(torch.ones((out_features, in_features), dtype=torch.int8), requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.ones((out_features), dtype=torch.int32), requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class for quantized linear layers. Please use Linear8 or Linear16 instead.")



class Linear8(LinearBase):
    """Linear8 is the Linear layer which has 8-bit requant output
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "integer inference (all int computation)"
        assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"

        if self.backend == "torch":
            return self._forward_torch(x)

        if self.backend == "scalar":
            return _C.linear8_scalar(
                x, self.weight, self.bias,
                self.wgt_zeros, self.req_multiplier, self.req_shiftmount
                )

        if self.backend == "simd":
            return _C.linear8_simd(
                x, self.weight, self.bias,
                self.wgt_zeros, self.req_multiplier, self.req_shiftmount
                )

        if self.backend == "check":
            ref = self._forward_torch(x)
            simd = _C.linear8_simd(
                x, self.weight, self.bias,
                self.wgt_zeros, self.req_multiplier, self.req_shiftmount
                )
            assert torch.equal(ref, simd), "SIMD backend result does not match reference!"
            return ref
        
    
    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        "integer inference (all int computation)"
        assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"
        '''INT8xINT8 --> INT32 MatMul'''
        out = torch.matmul(x.int(), self.weight.int().t())
        
        if self.bias is not None:
            '''bias addition'''
            out += self.bias
        
        if self.asym:
            '''Zero-point correction for asymmetric quantization'''
            out -= torch.sum(x.int(),dim=-1,keepdim=True).to(torch.int32) * self.wgt_zeros.to(torch.int32).reshape(1,1,-1)

        '''Re-Quantization'''
        out = (out.long().mul(self.req_multiplier.long())>>32).int() # MULH
        out = Rshift(out, self.req_shiftmount)
        out = out.clamp(min=-127,max=127).to(torch.int8)
        return out
    
    
    @staticmethod
    def copy_param(quant_linear):
        new_linear = Linear8(quant_linear.in_features, quant_linear.out_features, bias=quant_linear.bias is not None, asym=quant_linear.asym)
        new_linear.weight.data = quant_linear.weight.data.clone().detach()
        if quant_linear.bias is not None:
            new_linear.bias.data = quant_linear.bias.data.clone().detach()
        new_linear.req_multiplier.data = quant_linear.req_multiplier.data.clone().detach()
        new_linear.req_shiftmount.data = quant_linear.req_shiftmount.data.clone().detach()
        if quant_linear.asym:
            new_linear.wgt_zeros.data = quant_linear.wgt_zeros.data.clone().detach()
        return new_linear




class Linear32(LinearBase):
    """Linear32 is the Linear layer which has 32-bit output, and requant is left shift
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "integer inference (all int computation)"
        assert x.dtype == torch.int8, "Input activation must be int8 in integer inference mode!"
        '''INT8xINT8 --> INT32 MatMul'''
        if x.device.type == 'cuda':
            out = int8_linear_matmul(x.contiguous(), self.weight.contiguous())
        else: # using cpu
            out = torch.matmul(x.int(), self.weight.int().t())
        
        if self.bias is not None:
            '''bias addition'''
            out += self.bias
        
        if self.asym:
            '''Zero-point correction for asymmetric quantization'''
            out -= torch.sum(x.int(),dim=-1,keepdim=True).to(torch.int32) * self.wgt_zeros.to(torch.int32).reshape(1,1,-1)
        

        out = (out.long().mul(self.req_multiplier.long())>>32).int() # MULH
        out = (out << self.req_shiftmount).clamp(min=-2147483647,max=2147483647).to(torch.int32)

        return out
    
    @staticmethod
    def copy_param(quant_linear):
        new_linear = Linear32(quant_linear.in_features, quant_linear.out_features, bias=quant_linear.bias is not None, asym=quant_linear.asym)
        new_linear.weight.data = quant_linear.weight.data.clone().detach()
        if quant_linear.bias is not None:
            new_linear.bias.data = quant_linear.bias.data.clone().detach()
        new_linear.req_multiplier.data = quant_linear.req_multiplier.data.clone().detach()
        new_linear.req_shiftmount.data = quant_linear.req_shiftmount.data.clone().detach()
        if quant_linear.asym:
            new_linear.wgt_zeros.data = quant_linear.wgt_zeros.data.clone().detach()
        return new_linear


def set_linear8_backend(model, backend):
    for module in model.modules():
        if isinstance(module, Linear8):
            module.backend = backend
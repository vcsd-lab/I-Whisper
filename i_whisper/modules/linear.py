import torch
from torch import nn
from bitsandbytes.functional import int8_linear_matmul

from .functional import Rshift



class LinearBase(nn.Linear):
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

        '''Quantization Hyper-parameters (Ranges of Parameters)'''
        self.act_range = 127    # Signed, 2**(self.act_bit-1)-1

        '''Quantization Parameters initialization'''
        if self.asym:
            self.register_buffer('wgt_zeros', torch.zeros(out_features))
        self.req_mulop = torch.nn.Parameter(torch.ones((out_features), dtype=torch.uint32), requires_grad=False)
        self.req_shamt = torch.nn.Parameter(torch.zeros((out_features), dtype=torch.int8), requires_grad=False)
        
        '''weight re-initialization for quantization'''
        self.weight = nn.Parameter(torch.ones((out_features, in_features), dtype=torch.int8), requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(torch.ones((out_features), dtype=torch.int32), requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class for quantized linear layers. Please use Linear8 or Linear16 instead.")


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



class Linear8(LinearBase):
    """Linear8 is the Linear layer which has 8-bit requant output
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

        '''Re-Quantization'''
        out = (out.long().mul(self.req_mulop.long())>>32).int() # MULH
        out = Rshift(out, self.req_shamt)
        out = out.clamp(min=-2**7+1,max=2**7-1).to(torch.int8)
        return out
    
    @staticmethod
    def copy_param(quant_linear):
        new_linear = Linear8(quant_linear.in_features, quant_linear.out_features, bias=quant_linear.bias is not None, asym=quant_linear.asym)
        new_linear.weight.data = quant_linear.weight.data.clone().detach()
        if quant_linear.bias is not None:
            new_linear.bias.data = quant_linear.bias.data.clone().detach()
        new_linear.req_mulop.data = quant_linear.req_multiplier.data.clone().detach()
        new_linear.req_shamt.data = quant_linear.req_shiftmount.data.clone().detach()
        if quant_linear.asym:
            new_linear.wgt_zeros.data = quant_linear.wgt_zeros.data.clone().detach()
        return new_linear



class Linear16(LinearBase):
    """Linear16 is the Linear layer which has 16-bit requant output
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

        '''Re-Quantization'''
        out = (out.long().mul(self.req_mulop.long())>>32).int() # MULH
        out = Rshift(out, self.req_shiftmount)
        out = out.clamp(min=-2**15+1,max=2**15-1).to(torch.int16)
        return out



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
        

        out = (out.long().mul(self.req_mulop.long())>>32).int() # MULH
        out = (out << self.req_shamt).clamp(min=-2**31+1,max=2**31-1).to(torch.int32)

        return out
    
    @staticmethod
    def copy_param(quant_linear):
        new_linear = Linear32(quant_linear.in_features, quant_linear.out_features, bias=quant_linear.bias is not None, asym=quant_linear.asym)
        new_linear.weight.data = quant_linear.weight.data.clone().detach()
        if quant_linear.bias is not None:
            new_linear.bias.data = quant_linear.bias.data.clone().detach()
        new_linear.req_mulop.data = quant_linear.req_multiplier.data.clone().detach()
        new_linear.req_shamt.data = quant_linear.req_shiftmount.data.clone().detach()
        if quant_linear.asym:
            new_linear.wgt_zeros.data = quant_linear.wgt_zeros.data.clone().detach()
        return new_linear
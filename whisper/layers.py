from typing import Optional

import torch
import torch.nn as nn

class RoundWithGradient(torch.autograd.Function):   #對小數第一位四捨五入 往0的方向   -0.4 => 0
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g

class ActQuantizer(nn.Module):
    def __init__(self, num_bits=8, asym:Optional[bool]=False, QMODE:Optional[str]='MAX'):
        super(ActQuantizer, self).__init__()
        self.num_bits = num_bits
        self.calibrate= False
        self.quantize = False
        self.asym = asym
        self.QMODE = QMODE

        self.ActsScale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.ZeroPoint = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        if self.asym:
            self.Qmin = 0.0
            self.Qmax = 2**self.num_bits - 1
        else:
            self.Qmin = -2**(self.num_bits - 1)
            self.Qmax = 2**(self.num_bits - 1)
        
        self.init_calibre = True
        self.calibre_max = None
        self.calibre_min = None

    def sgn(self, x):
        x = RoundWithGradient.apply(x)
        return x

    def forward(self, x, preln=False):
        if self.quantize:
            if preln:
                return x.div(self.ActsScale).round().sub(self.ZeroPoint).clamp(self.Qmin, self.Qmax)
            else:
                return x.div(self.ActsScale).round().sub(self.ZeroPoint).clamp(self.Qmin, self.Qmax).add(self.ZeroPoint).mul(self.ActsScale)
            
        else:
            if self.calibrate:
                cur_max = None
                cur_min = None
                if self.QMODE == 'MAX':
                    cur_max = x.max()
                    cur_min = x.min()
                elif self.QMODE == '0.999':
                    cur_max = torch.quantile(x, 0.999, interpolation='lower')
                    cur_min = torch.quantile(x, 0.001, interpolation='higher')
                
                if self.init_calibre:
                    self.calibre_max = cur_max
                    self.calibre_min = cur_min
                else:
                    self.calibre_max = torch.maximum(self.calibre_max, cur_max)
                    self.calibre_min = torch.minimum(self.calibre_min, cur_min)

            return x
                    
    def open_quant(self):
        if self.quantize or self.calibre_max is None or self.calibre_min is None:
            self.quantize = True
            return
        self.calibrate = False
    
        if self.asym:
            self.ActsScale.data = (self.calibre_max - self.calibre_min) / (self.Qmax - self.Qmin)
            self.ZeroPoint.data = torch.round(self.calibre_min / self.ActsScale.data)
        else:
            self.ActsScale.data = torch.maximum(torch.abs(self.calibre_max), torch.abs(self.calibre_min)) / self.Qmax
    
    def close_quant(self):
        self.quantize = False
    
    def open_calibrate(self):
        self.calibrate = True
        self.init_calibre = True
        self.calibre_max = None
        self.calibre_min = None
    
    def close_calibrate(self):
        self.calibrate = False
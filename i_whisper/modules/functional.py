import torch
from torch import nn
from bitsandbytes.functional import int8_linear_matmul



def Rshift(x, N):
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
    # assert x.dtype is torch.int32, "Rshift function only support int32 tensor!"

    try:
        if x.dtype is torch.int32:
            ROUNDING_BIAS = (1 << (N-1).to(torch.int32).clamp(-32,32)) - torch.signbit(x).to(torch.int32)
            return (x + ROUNDING_BIAS)>>(N.to(torch.int32).clamp(-32,32))
        elif x.dtype is torch.int64:
            ROUNDING_BIAS = (1 << (N-1).to(torch.int64).clamp(-64,64)) - torch.signbit(x).to(torch.int64)
            return (x + ROUNDING_BIAS)>>(N.to(torch.int64).clamp(-64,64))
    except:
        ROUNDING_BIAS = (1 << N-1) - torch.signbit(x).int()
        return (x + ROUNDING_BIAS)>>N



def Batch_Head_s8s8s32_GeMM(A: torch.Tensor, B: torch.Tensor):
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
        Out = A.to('cpu',torch.int32) @ B.to('cpu',torch.int32).transpose(-1,-2)
    return Out


def Batch_Head_u8s8s32_GeMM(A: torch.Tensor, B: torch.Tensor):
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
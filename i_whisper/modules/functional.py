import torch
from torch import nn
from bitsandbytes.functional import int8_linear_matmul
import math



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
    if x.dtype is not torch.int32:
        raise NotImplementedError(
            "Rshift only supports int32 tensor!"
        )

    if not isinstance(N, torch.Tensor):
        N = torch.tensor(
            N,
            dtype=torch.int32,
            device=x.device,
        )
    else:
        N = N.to(
            dtype=torch.int32,
            device=x.device,
        )

    shift = N.clamp(0, 32)

    rounding_shift = (shift - 1).clamp(min=0)

    ROUNDING_BIAS = (
        (1 << rounding_shift)
        - torch.signbit(x).to(torch.int32)
    )

    output = (
        x + ROUNDING_BIAS
    ) >> shift

    # torch.where broadcasts condition to x/output shape
    output = torch.where(
        shift == 0,
        x,
        output,
    )

    return output



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


def integer_rsqrt_initial_guess(
    x: torch.Tensor,
    lut: torch.Tensor,
    frac_bits: int = 20,
    lut_bits: int = 8,
):
    """
    x:
        non-negative INT32 integer value

    lut:
        shape [2, 2**lut_bits]
        Q(frac_bits) initial rsqrt approximation

    return:
        Q(frac_bits) integer approximation of 1/sqrt(x)
    """

    assert x.dtype == torch.int32
    assert lut.dtype == torch.int32

    zero_mask = (x == 0)

    # --------------------------------------------------
    # exponent e = floor(log2(x))
    # --------------------------------------------------

    bit_length = int_bit_length(x)

    e = (bit_length - 1).clamp(min=0)

    parity = e & 1
    k = e >> 1

    # --------------------------------------------------
    # Normalize x into:
    #
    #   m_q30 = m * 2^30
    #
    # where 1 <= m < 2.
    # --------------------------------------------------

    shift = 30 - e

    x64 = x.to(torch.int64)

    m_q30 = torch.where(
        shift >= 0,
        x64 << shift,
        x64 >> (-shift),
    )

    # m_q30:
    #
    # bit 30       = implicit leading 1
    # bits 29 ...  = fractional mantissa

    idx_shift = 30 - lut_bits

    idx = (
        m_q30 >> idx_shift
    ) & ((1 << lut_bits) - 1)

    idx = idx.to(torch.int64)
    parity_idx = parity.to(torch.int64)

    # Q20 approximation of:
    #
    # 1 / sqrt(m * 2^parity)
    #
    y = lut[parity_idx, idx]

    # Account for exponent 2*k
    y = Rshift(y, k)

    # Define variance==0 semantics explicitly.
    y = torch.where(
        zero_mask,
        torch.full_like(y, 1 << frac_bits),
        y,
    )

    return y.to(torch.int32)


def build_rsqrt_lut(
    lut_bits=8,
    frac_bits=20,
):
    n = 1 << lut_bits

    lut = torch.empty(
        (2, n),
        dtype=torch.int32
    )

    for parity in range(2):
        for idx in range(n):

            # center of LUT interval
            m = 1.0 + (idx + 0.5) / n

            value = 1.0 / math.sqrt(
                m * (2 ** parity)
            )

            lut[parity, idx] = round(
                value * (1 << frac_bits)
            )

    return lut



def int_bit_length(x: torch.Tensor) -> torch.Tensor:
    """
    Bit length of non-negative integer tensor.

    0 -> 0
    1 -> 1
    2 -> 2
    3 -> 2
    4 -> 3
    ...

    Equivalent to:
        floor(log2(x)) + 1, x > 0
        0,                  x == 0
    """

    assert not x.is_floating_point()
    assert torch.all(x >= 0)

    # Use int64 only as an integer container.
    # No floating-point arithmetic is involved.
    y = x.to(torch.int64)

    bitpos = torch.zeros_like(y)

    # Binary-search style CLZ / MSB search.
    for shift in (16, 8, 4, 2, 1):
        upper = y >> shift
        has_upper = upper != 0

        y = torch.where(
            has_upper,
            upper,
            y,
        )

        bitpos = bitpos + has_upper.to(torch.int64) * shift

    bit_length = bitpos + (x != 0).to(torch.int64)

    return bit_length.to(torch.int32)




# def Int32_To_FP32_Bits(x: torch.Tensor) -> torch.Tensor:
#     """
#     Emulate positive int32 -> float32 conversion,
#     but return the IEEE-754 bit pattern as int32.

#     No floating-point tensor is created.
#     """

#     assert x.dtype == torch.int32
#     assert torch.all(x >= 0)

#     x64 = x.to(torch.int64)

#     zero = (x64 == 0)

#     # e = floor(log2(x))
#     bit_length = int_bit_length(x)
#     e = (bit_length - 1).to(torch.int64)

#     # --------------------------------------------------
#     # Case A: e <= 23
#     #
#     # Integer can be represented exactly by FP32.
#     # Shift its leading 1 to bit 23.
#     # --------------------------------------------------

#     left_shift = (23 - e).clamp(min=0)

#     sig_exact = x64 << left_shift

#     # --------------------------------------------------
#     # Case B: e > 23
#     #
#     # Need to discard low bits and emulate
#     # round-to-nearest-even used by FP32 conversion.
#     # --------------------------------------------------

#     right_shift = (e - 23).clamp(min=0)

#     sig_round = x64 >> right_shift

#     remainder = x64 - (sig_round << right_shift)

#     half_shift = (right_shift - 1).clamp(min=0)

#     halfway = torch.where(
#         right_shift > 0,
#         torch.ones_like(x64) << half_shift,
#         torch.zeros_like(x64),
#     )

#     round_up = (
#         (right_shift > 0)
#         & (
#             (remainder > halfway)
#             | (
#                 (remainder == halfway)
#                 & ((sig_round & 1) != 0)
#             )
#         )
#     )

#     sig_round = sig_round + round_up.to(torch.int64)

#     # Select exact or rounded significand.
#     significand = torch.where(
#         e <= 23,
#         sig_exact,
#         sig_round,
#     )

#     # Rounding can turn 1.111... into 10.000...
#     carry = significand >= (1 << 24)

#     significand = torch.where(
#         carry,
#         significand >> 1,
#         significand,
#     )

#     e = e + carry.to(torch.int64)

#     exponent = e + 127

#     mantissa = significand & 0x7FFFFF

#     bits = (
#         (exponent << 23)
#         | mantissa
#     )

#     bits = torch.where(
#         zero,
#         torch.zeros_like(bits),
#         bits,
#     )

#     return bits.to(torch.int32)

def Int32_To_FP32_Bits_Trunc(x: torch.Tensor):
    """
    Approximate positive int32 -> IEEE754 FP32 bit pattern.

    Mantissa is truncated instead of IEEE round-to-nearest-even.
    Integer-only.
    """
    assert x.dtype == torch.int32
    assert torch.all(x >= 0)

    zero = (x == 0)

    x64 = x.to(torch.int64)

    bit_length = int_bit_length(x)
    e = (bit_length - 1).clamp(min=0).to(torch.int64)

    # Align leading 1 to bit 23.
    left_shift  = (23 - e).clamp(min=0)
    right_shift = (e - 23).clamp(min=0)

    significand = torch.where(
        e <= 23,
        x64 << left_shift,
        x64 >> right_shift,    # truncate low bits
    )

    exponent = e + 127
    mantissa = significand & 0x7FFFFF

    bits = (exponent << 23) | mantissa

    bits = torch.where(
        zero,
        torch.zeros_like(bits),
        bits,
    )

    return bits.to(torch.int32)


# def FP32_Bits_To_Int32(
#     bits: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Interpret positive IEEE-754 FP32 bit patterns and
#     truncate their numerical values to int32.

#     Equivalent in intended range to:

#         bits.view(torch.float32).to(torch.int32)

#     without creating FP tensors.
#     """

#     bits64 = bits.to(torch.int64) & 0xFFFFFFFF

#     sign = (bits64 >> 31) & 1

#     exponent_raw = (
#         bits64 >> 23
#     ) & 0xFF

#     fraction = bits64 & 0x7FFFFF

#     # Our rsqrt seed should always be positive.
#     assert torch.all(sign == 0)

#     # Handle zero/subnormal as zero for integer conversion.
#     is_normal = (
#         (exponent_raw != 0)
#         & (exponent_raw != 0xFF)
#     )

#     e = exponent_raw - 127

#     significand = (
#         (1 << 23) | fraction
#     )

#     left_shift = (
#         e - 23
#     ).clamp(min=0)

#     right_shift = (
#         23 - e
#     ).clamp(min=0)

#     integer_value = torch.where(
#         e >= 23,
#         significand << left_shift,
#         significand >> right_shift,
#     )

#     integer_value = torch.where(
#         (e >= 0) & is_normal,
#         integer_value,
#         torch.zeros_like(integer_value),
#     )

#     return integer_value.to(torch.int32)

def FP32_Bits_To_Int32_Trunc(bits: torch.Tensor):
    bits64 = bits.to(torch.int64) & 0xFFFFFFFF

    exponent_raw = (bits64 >> 23) & 0xFF
    fraction = bits64 & 0x7FFFFF

    e = exponent_raw - 127

    significand = (
        (1 << 23) | fraction
    )

    left_shift = (e - 23).clamp(min=0)
    right_shift = (23 - e).clamp(min=0)

    y = torch.where(
        e >= 23,
        significand << left_shift,
        significand >> right_shift,
    )

    y = torch.where(
        e >= 0,
        y,
        torch.zeros_like(y),
    )

    return y.to(torch.int32)

def Fast_Inverse_Square_Root(x, offset, Newton_iter: int = 0, overflow_prevent_offset: int = 0):
    '''fast inverse square root in flaot32 format'''
    assert x.dtype == torch.int32
    assert torch.all(x >= 0)
    
    # zero_pos = (x == 0)
    # i = x.type(torch.float32).view(dtype=torch.int32)
    i = Int32_To_FP32_Bits_Trunc(x) # convert int32 to float32 bits
    i = int(0x5f3759df) - (i >> 1) # initial point with magic number

    '''convert potential float32 in int32 to fixed point q12.20 format'''
    # try new convert method
    # add offset to exponent directly then convert to fixed point (int32)
    y = FP32_Bits_To_Int32_Trunc(i + (offset<<23))
    # y = (i + (offset<<23))
    # y = y.view(torch.float32)
    # y = y.to(torch.int32)
    # without this, inverse square root of zero is infinity, here we set it to one directly
    # this means do not need normalize when input (variation) is zero

    
    '''iteration of Newton's method (in fixed point q12.20 format)'''
    three = torch.tensor(3<<offset, dtype=torch.int32) # 3 in fixed point q12.20 format
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






def LUT_Rsqrt_Initial_Guess(
    x: torch.Tensor,
    lut: torch.Tensor,
    frac_bits: int,
    lut_bits: int,
):
    assert x.dtype == torch.int32
    assert lut.dtype == torch.int32

    zero_mask = (x == 0)

    bit_length = int_bit_length(x)

    # e = floor(log2(x))
    e = (bit_length - 1).clamp(min=0)

    # e = 2*k + parity
    parity = e & 1
    k = e >> 1

    # Normalize x to Q30:
    #
    #     1 <= m < 2
    #
    shift = 30 - e

    x64 = x.to(torch.int64)

    m_q30 = torch.where(
        shift >= 0,
        x64 << shift,
        x64 >> (-shift),
    )

    # Remove implicit leading 1.
    fraction = m_q30 - (1 << 30)

    # Take the top lut_bits of fractional mantissa.
    idx = fraction >> (30 - lut_bits)

    idx = idx.to(torch.int64)
    parity = parity.to(torch.int64)

    # LUT already stores Q(frac_bits)
    y = lut[parity, idx]

    # Account for exponent:
    #
    # 1/sqrt(x) = LUT(m, parity) * 2^-k
    y = Rshift(y, k)

    # variance == 0 → normalization factor 1.0
    y = torch.where(
        zero_mask,
        torch.full_like(y, 1 << frac_bits),
        y,
    )

    return y.to(torch.int32)



def LUT_Based_Inverse_Square_Root(
    x,
    lut,
    offset,
    lut_bits=6,
    Newton_iter=0,
):
    y = LUT_Rsqrt_Initial_Guess(
        x,
        lut,
        frac_bits=offset,
        lut_bits=lut_bits,
    )

    three = torch.tensor(
        3 << offset,
        dtype=torch.int32,
        device=x.device,
    )

    def Newton_iteration(y, x):
        t = Rshift(
            x.mul(y).int(),
            offset,
        ).mul(y)

        y = Rshift(
            y.mul(three - t).int(),
            offset + 1,
        )

        return y

    for _ in range(Newton_iter):
        y = Newton_iteration(y, x)

    return y
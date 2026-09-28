import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
from math import log2
from os import path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from .utils import exact_div
import struct

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

DUMPPARAM = False # Dump the parameters to a file for testing
DECIMAL_BITS = 10  # The FFT is quantized to int32, so we need to shift right by 10 bits

# print the error between float and int FFT
SHOW_ERROR = False

# DUMP the parameters for testing
DUMP_TWIDDLE = False # Dump the twiddle factors to a file for testing
DUMP_TWIDDLE_LIST = []
DUMP_FILTER = False # Dump the mel filterbank & window filter to a file for testing
DUMP_LOGLUT = False # Dump the log lookup table to a file for testing

# record the data as golden for testing
INPUT_GOLDEN = False # Generate golden input for testing
WINDOW_GOLDEN = False # Generate golden windowed for testing
FFT_GOLDEN = False # Generate golden parameters for testing
SPECTRUM_GOLDEN = False # Generate golden spectrum for testing
MEL_GOLDEN = False # Generate golden mel spectrogram for testing
LOGMEL_GOLDEN = False # Generate golden log mel spectrogram for testing

def load_audio(file: str, sr: int = SAMPLE_RATE, raw: bool = False) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    if raw:
        return np.frombuffer(out, np.int16).flatten().astype(np.int16)
    else:
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


import librosa
def get_mel_filters(device, n_mels: int = 80, n_fft: int = 512):
    mel_filter = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=n_mels)
    return torch.tensor(mel_filter).to(device)

def fft(x):
    N = x.shape[-1]  # get fft tap size

    if N <= 1:
        return x
    
    if N == 4:
        x = x.to(torch.complex64)
    
    if FFT_GOLDEN and N >= 4:
        with open("saved_parameters/float/fft_input_tap_"+str(N), "w") as f:
            for i in range(x.shape[1]):
                f.write(f"{x[0,i].real}\n")

    # 偶數部分和奇數部分的拆分
    even = fft(x[:,::2])  # 偶數索引部分
    odd = fft(x[:,1::2])   # 奇數索引部分

    if FFT_GOLDEN and N>=4:
        with open("saved_parameters/float/fft_even_tap_"+str(N), "w") as f:
            for i in range(even.shape[1]):
                f.write(f"{even[0,i].real} {even[0,i].imag}\n")
        with open("saved_parameters/float/fft_odd_tap_"+str(N), "w") as f:
            for i in range(odd.shape[1]):
                f.write(f"{odd[0,i].real} {odd[0,i].imag}\n")

    if N > 4:
        # 合併步驟
        T = torch.exp(-2j * torch.pi * torch.arange(N // 2) / N).unsqueeze(0)  # 旋轉因子（twiddle factors）
        T.real.round_(decimals=7)  # 將實部四捨五入到小數點後7位
        T.imag.round_(decimals=7)
        
        
        if DUMP_TWIDDLE:
            if not N in DUMP_TWIDDLE_LIST:
                DUMP_TWIDDLE_LIST.append(N)
                with open("saved_parameters/float/fp_twiddle_factors.hpp", "a") as f:
                    f.write("\n/**\n")
                    f.write(f" * Function twiddle_factors_{N}\n")
                    f.write(" * This Function returns twiddle factors in vector<complex<float> >.\n")
                    f.write(f" * For {N} taps FFT, size of twiddle factor is {N//2}\n")
                    f.write(f" * Thus, this function return the vector with length = {N//2}.\n")
                    f.write(" */\n")
                    f.write(f"vector<complex<float> > twiddle_factors_{N}() "+"{\n")
                    f.write(f"    vector<complex<float> >twiddle_factor = "+"{\n")
                    for i in range(T.shape[1]):
                        f.write(f"        complex<float>({T[0,i].real}, {T[0,i].imag})")
                        if i != T.shape[1]-1:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.write("    };\n")
                    f.write("    return twiddle_factor;\n")
                    f.write("}\n")

        odd = T * odd  # 使用 twiddle factors 乘以奇數部分

    elif N == 4:
        real = odd[:, 1].real.detach().clone()  # 將奇數部分的第二個元素翻轉，reduce some twiddle calculation
        imag = odd[:, 1].imag.detach().clone()
        odd[:, 1].real = -imag
        odd[:, 1].imag = real  # 將奇數部分的第二個元素翻轉，reduce some twiddle calculation

    twiddled = torch.cat([even + odd, even - odd], dim=-1)  # 合併偶數和奇數部分
    
    if FFT_GOLDEN and N >= 4:
        with open("saved_parameters/float/fft_twiddled_tap_"+str(N), "w") as f:
            for i in range(twiddled.shape[1]):
                f.write(f"{twiddled[0,i].real} {twiddled[0,i].imag}\n")
    
    
    return twiddled


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    magnitudes = stft[..., :-1].abs() ** 2

    filters = get_mel_filters(audio.device, n_mels, N_FFT)
    mel_spec = filters.to(magnitudes) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = log_spec/4.0 + 1.0
    return log_spec



def int_fft(x):
    N = x.shape[-2]

    # if FFT_GOLDEN:
    #     with open("saved_parameters/int/fft_input_tap_"+str(N), "w") as f:
    #         for i in range(x.shape[1]):
    #             f.write(f"{x[0,i,0]}\n")

    if N <= 1:
        return x

    # 偶數部分和奇數部分的拆分
    even = int_fft(x[:, ::2, :])  # 偶數索引部分
    odd = int_fft(x[:, 1::2, :])   # 奇數索引部分

    # if FFT_GOLDEN and N>=4:
    #     with open("saved_parameters/int/fft_even_tap_"+str(N), "w") as f:
    #         for i in range(even.shape[1]):
    #             f.write(f"{even[0,i,0]} {even[0,i,1]}\n")
    #     with open("saved_parameters/int/fft_odd_tap_"+str(N), "w") as f:
    #         for i in range(odd.shape[1]):
    #             f.write(f"{odd[0,i,0]} {odd[0,i,1]}\n")
    # 合併步驟
    if N > 4:
        TWIDDLE_BITS = 7
        T = torch.exp(2j * torch.pi * torch.arange(N // 2) / N).unsqueeze(0)  # 旋轉因子（twiddle factors）
        T = torch.view_as_real(T)  # 將複數轉換為實部和虛部
        T = T.mul(2**(TWIDDLE_BITS-1)).round().to(torch.int32) # 將旋轉因子轉換為 int32，這樣可以避免浮點數計算的精度問題



        if DUMP_TWIDDLE:
            if not N in DUMP_TWIDDLE_LIST:
                PATH = os.path.join("DEMO/AudioPreProcess/DATA/PARAM", f"twiddle_factors.hpp")
                DUMP_TWIDDLE_LIST.append(N)
                with open(PATH, "a") as f:
                    f.write("\n/**\n")
                    f.write(f" * List twiddle_factors_{N}\n")
                    f.write(f" * This List is twiddle factors in List<IntComplex>.\n")
                    f.write(f" * For {N} taps FFT, size of twiddle factor is {N//2}\n")
                    f.write(f" * and each IntComplex32 Stores 1 Complex.\n")
                    f.write(f" * Thus, this List is a vector with length = {N//2}.\n")
                    f.write(f" */\n")

                    f.write(f"const int16_t twiddle_factors_{N}[{N//2}][2]"+" __attribute__((section(\".data.twiddle"+f"{N:03d}"+"\"))) = {\n")
                    for i in range(T.shape[1]):
                        f.write("    {"+f"{T[0,i,0]}, {T[0,i,1]}"+"}")
                        if i != T.shape[1]-1:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.write("};\n")




    def complex_mul(a, b):
        """複數乘法，將實部和虛部分開計算"""
        assert a.shape[-1] == 2 and b.shape[-1] == 2, "Input tensors must have shape (..., 2) for complex numbers"
        result_real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        result_imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return torch.cat([result_real.unsqueeze(-1), result_imag.unsqueeze(-1)], dim=-1).bitwise_right_shift(TWIDDLE_BITS-1)
    
    

    if N > 4:
        odd = complex_mul(T, odd)  # 使用 twiddle factors 乘以奇數部分

    if N == 4:
        odd[:,1,:] = odd[:,1,:].flip(-1)  # 將奇數部分的第二個元素翻轉，reduce some twiddle calculation
        odd[:,1,0].mul_(-1)

    twiddled = torch.cat([even + odd, even - odd], dim=-2)  # 合併偶數和奇數部分

    # if FFT_GOLDEN and N >= 4:
    #     with open("saved_parameters/int/fft_twiddled_tap_"+str(N), "w") as f:
    #         for i in range(twiddled.shape[1]):
    #             f.write(f"{twiddled[0,i,0]} {twiddled[0,i,1]}\n")


    if SHOW_ERROR:
        float_even = torch.view_as_complex(even.float().div(2**DECIMAL_BITS))
        float_odd = torch.view_as_complex(odd.float().div(2**DECIMAL_BITS))
        float_T = torch.exp(-2j * torch.pi * torch.arange(N // 2) / N).unsqueeze(0)
        float_twiddled = torch.cat([float_even + float_T * float_odd, float_even - float_T * float_odd],dim=-1)
        dequant_twiddled = torch.view_as_complex(twiddled.float().div(2**DECIMAL_BITS))
        comparison = torch.allclose(float_twiddled, dequant_twiddled, atol=1e-2)
        if not comparison:
            print("Taps now is", N,", FFT Quantization Error larger than 1e-2")
    return twiddled


class PreProcessor:
    def __init__(self, n_mels: int = 80, n_fft = 512, hop_length = 160, param_init = True, device: Optional[Union[str, torch.device]] = None):
        self.device = device
        self.n_mels = n_mels
        self.hop_length = hop_length
        assert 2**log2(n_fft) == n_fft, "n_fft must be power of 2"
        self.n_fft = n_fft
        self.lut_built = False

        self.base_offset = 10
        self.window_shmnt = 8
        self.twiddle_offset = 7
        self.mel_offset = 8
        
        self.twiddle = {}
        self.out_table = None
        self.idx_table = None
        if param_init:
            ''' Build window function
            '''
            window = torch.hann_window(self.n_fft).to(self.device)
            window = window.mul(2**self.window_shmnt).round().clamp(0,2**self.window_shmnt-1).to(torch.uint8)
            self.window = window
            
            ''' Build Twiddle Factors for FFT
            '''
            for i in np.arange(3,int(log2(self.n_fft))+1):
                N = int(2**i)
                T = torch.exp(2j * torch.pi * torch.arange(N // 2, device=self.device) / N).unsqueeze(0)
                T = torch.view_as_real(T)
                T = T.mul(2**(self.twiddle_offset-1)).round().to(torch.int8)
                self.twiddle[N] = T # key: N taps, value: twiddle factors tensor
            
            ''' Build Mel Filterbank
            '''
            filters = get_mel_filters(self.device, n_mels=self.n_mels, n_fft=self.n_fft)
            self.filters_shmnt = torch.log2(filters.max()).round().abs() + self.mel_offset # Scale to fit in int32
            self.filters = filters.mul(2**self.filters_shmnt).round().clamp(0,2**self.mel_offset-1).to(torch.uint8)  # Quantize to int8
        else:
            self.load_param()
        

    def log_lut_build(self, result_alpha:float):
        out_table = {}
        table_index = []
        full_out_table = torch.arange(-128, 128, dtype=torch.float32, device=self.device) # [-128,127]
        full_idx_table = torch.arange(-127.5, 127.5, 1.0, dtype=torch.float32, device=self.device) # rounding threshold of [-128,127]
        full_idx_table /= result_alpha # inverse re-map compute { output = round(input * alpha) }
        full_idx_table = (full_idx_table-1)*4 # inverse re-map compute { output = input/4 + 1 }
        full_idx_table = 10**full_idx_table # inverse re-map compute { output = log10(input) }
        full_idx_table = full_idx_table.mul(2**self.base_offset).round().to(torch.int32).clamp(0,2**16) # re-map to input range fixed-point
        for i in reversed(range(full_idx_table.shape[0])):
            '''
            remove duplicate entries in the lookup table
            because log function has many-to-one mapping after quantization
            e.g. log(1.23) and log(1.234) may be both quantized to 123 after scaling and rounding
            thus we only keep one of them in the lookup table
            '''
            if not full_idx_table[i].item() in out_table:
                out_table[full_idx_table[i].item()] = full_out_table[i+1].item()
                table_index.append(full_idx_table[i].item())
        self.idx_table = torch.tensor(table_index).sort().values # sorted index table
        self.out_table = out_table # key: input (idx), value: output
        self.lut_built = True
        return self.lut_built

        
    def save_param(self, root:str = "CKPT"):
        """ Param of audio PreProcessor will save in numpy's npy format
        reason is that npz is easy to read in python and c++
        """
        if not os.path.exists(root):
            os.mkdir(root)
        PATH = os.path.join(root, f"audio_preprocessor_params.npy")
        SAVE_DICT = {}
        SAVE_DICT['n_mels'] = self.n_mels
        SAVE_DICT['n_fft'] = self.n_fft
        SAVE_DICT['hop_length'] = self.hop_length
        SAVE_DICT['base_offset'] = self.base_offset
        SAVE_DICT['window_shmnt'] = self.window_shmnt
        SAVE_DICT['twiddle_offset'] = self.twiddle_offset
        SAVE_DICT['mel_offset'] = self.mel_offset
        SAVE_DICT['lut_built'] = self.lut_built
        SAVE_DICT['window'] = self.window.cpu().numpy()
        SAVE_DICT['twiddle'] = self.twiddle
        SAVE_DICT['filters_shmnt'] = self.filters_shmnt
        SAVE_DICT['filters'] = self.filters.cpu().numpy()
        if self.lut_built:
            SAVE_DICT['out_table'] = self.out_table
            SAVE_DICT['idx_table'] = self.idx_table.cpu().numpy()
        np.save(PATH, SAVE_DICT)
        return
        
    def load_param(self):
        if not os.path.exists("CKPT"):
            raise FileNotFoundError("No CKPT folder found!")
        PATH = os.path.join("CKPT", f"audio_preprocessor_params.npy")
        LOAD_DICT = np.load(PATH, allow_pickle=True).item()
        # CHECK NUMBER OF KEYS
        missing_key = []
        if len(LOAD_DICT.keys()) != 14:
            # find which key is missing
            for key in ['n_mels', 'n_fft', 'hop_length', 'lut_built',
                        'base_offset', 'window_shmnt', 'twiddle_offset', 'mel_offset',
                        'window', 'twiddle', 'filters_shmnt', 'filters', 'out_table', 'idx_table']:
                if not key in LOAD_DICT:
                    missing_key.append(key)
            print("Missing keys in loaded PreProcessor parameters:", missing_key)
        else:
            self.n_mels = LOAD_DICT['n_mels']
            self.n_fft = LOAD_DICT['n_fft']
            self.hop_length = LOAD_DICT['hop_length']
            self.base_offset = LOAD_DICT['base_offset']
            self.window_shmnt = LOAD_DICT['window_shmnt']
            self.twiddle_offset = LOAD_DICT['twiddle_offset']
            self.mel_offset = LOAD_DICT['mel_offset']
            self.lut_built = LOAD_DICT['lut_built']
            self.window = torch.from_numpy(LOAD_DICT['window']).to(self.device)
            self.twiddle = LOAD_DICT['twiddle']
            self.filters_shmnt = LOAD_DICT['filters_shmnt']
            self.filters = torch.from_numpy(LOAD_DICT['filters']).to(self.device)
            try:
                self.out_table = LOAD_DICT['out_table']
                self.idx_table = torch.from_numpy(LOAD_DICT['idx_table']).to(self.device)
            except:
                pass
        return missing_key
    

    def int_fft(self, x):
        N = x.shape[-2]

        if N <= 1:
            return x

        even = self.int_fft(x[:, ::2, :])
        odd  = self.int_fft(x[:, 1::2, :])

        def complex_mul(a, b):
            """complex multiplication for tensors with last dimension size 2"""
            assert a.shape[-1] == 2 and b.shape[-1] == 2, "Input tensors must have shape (..., 2) for complex numbers"
            result_real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
            result_imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
            return torch.cat([result_real.unsqueeze(-1), result_imag.unsqueeze(-1)], dim=-1).bitwise_right_shift(self.twiddle_offset-1)
        
        if N > 4:
            odd = complex_mul(self.twiddle[N], odd)

        if N == 4:
            odd[:,1,:] = odd[:,1,:].flip(-1)
            odd[:,1,0].mul_(-1)

        twiddled = torch.cat([even + odd, even - odd], dim=-2)  # 合併偶數和奇數部分

        return twiddled
        

    def log_mel_spectrum(
            self,
            audio: Union[str, np.ndarray, torch.Tensor],
            padding: int = 0,
        ):
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))

        # TODO: some signal preprocess
        # 1. Pad by reflect at the beginning and end because STFT do this in default
        # 2. Unfold the audio into frames of N_FFT samples with HOP_LENGTH stride
        # 3. Apply a Hann window to each frame
        # 4. Compute the FFT of each frame

        # 1. Pad by reflect at the beginning and end
        signal_dim = audio.dim()
        extended_shape = [1] * (3 - signal_dim) + list(audio.size())
        pad = int(self.n_fft // 2)
        audio = F.pad(audio.view(extended_shape), [pad, pad], 'reflect')
        audio = audio.view(audio.shape[-signal_dim:]) # This is full precision input, reserved for testing

        # 2. Unfold the audio into frames of N_FFT samples with HOP_LENGTH stride
        unfolded_audio = audio.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        if audio.dtype != torch.int16:
            unfolded_audio = unfolded_audio.mul(2**self.base_offset).round().to(torch.int32)
        else:
            unfolded_audio = unfolded_audio.to(torch.int32) >> (15-self.base_offset)
    

        # 3. Apply a Hann window to each frame
        windowed_audio = (unfolded_audio * self.window.int().view([1]*(unfolded_audio.dim()-1)+[-1])).bitwise_right_shift(self.window_shmnt)
        # Add imag dimension (convert to complex number)
        windowed_audio = torch.cat([windowed_audio.unsqueeze(-1), torch.zeros_like(windowed_audio.unsqueeze(-1))], dim=-1)

        # 4. Compute the FFT of each frame
        stft_fft = self.int_fft(windowed_audio)[:,:self.n_fft//2+1,:]  # Transpose to (batch, n_frames, n_bins, 2) for complex numbers
        magnitudes = stft_fft[...,0].abs() ** 2 + stft_fft[...,1].abs() ** 2  # calculate magnitude from real and imaginary parts
        magnitudes = magnitudes.t()[..., :-1].bitwise_right_shift(self.base_offset) # Drop last frame

        mel_spec = (self.filters.int() @ magnitudes.int()).bitwise_right_shift(self.filters_shmnt.int()) # Project to Mel spectrogram

        if self.lut_built:
            output = torch.zeros_like(mel_spec, dtype=torch.int8, device=audio.device)
            for i in range(len(self.idx_table)):
                output[mel_spec > self.idx_table[i]] = self.out_table[self.idx_table[i].item()]
            return output
        else:
            mel_spec = mel_spec.float().div(2**self.base_offset)
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = log_spec/4.0 + 1.0
            return log_spec
    


def log_mel_spectrogram_fft(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    result_alpha: float = None,
    int_mode: Optional[bool] = False
):
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    
    
    window = torch.hann_window(N_FFT).to(audio.device)

    BASE = os.path.join('DEMO/AudioPreProcess/DATA')
    if not os.path.exists(BASE):
        os.makedirs(BASE)
    GOLDEN_BASE = os.path.join(BASE, 'GOLDEN')
    if not os.path.exists(GOLDEN_BASE):
        os.makedirs(GOLDEN_BASE)
    PARAM_BASE = os.path.join(BASE, 'PARAM')
    if not os.path.exists(PARAM_BASE):
        os.makedirs(PARAM_BASE)

    if SHOW_ERROR:
        stft_origin = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    # TODO: some signal preprocess
    # 1. Pad by reflect at the beginning and end because STFT do this in default
    # 2. Unfold the audio into frames of N_FFT samples with HOP_LENGTH stride
    # 3. Apply a Hann window to each frame
    # 4. Compute the FFT of each frame

    # 1. Pad by reflect at the beginning and end
    signal_dim = audio.dim()
    extended_shape = [1] * (3 - signal_dim) + list(audio.size())
    pad = int(N_FFT // 2)
    audio = F.pad(audio.view(extended_shape), [pad, pad], 'reflect')
    audio = audio.view(audio.shape[-signal_dim:]) # This is full precision input, reserved for testing

    # 2. Unfold the audio into frames of N_FFT samples with HOP_LENGTH stride
    unfolded_audio = audio.unfold(dimension=-1, size=N_FFT, step=HOP_LENGTH)
    if audio.dtype != torch.int16:
        unfolded_audio = unfolded_audio.mul(2**DECIMAL_BITS).round().to(torch.int32)
    
    NUM_OF_FRAMES = 10
    for i in range(NUM_OF_FRAMES):
        FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
        if not os.path.exists(FRAME_BASE):
            os.makedirs(FRAME_BASE)

    if INPUT_GOLDEN:
        for i in range(NUM_OF_FRAMES):
            FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
            PATH = os.path.join(FRAME_BASE, 'input_signal.S')
            with open(PATH, "w") as f:
                f.write("# Define Input signals\n")
                f.write(".section .data\n")
                f.write(".align 2\n")
                f.write(".global input_signal\n")
                f.write("input_signal:\n")
                for j in range(512):
                    hex_val = struct.unpack(">H", struct.pack(">h", unfolded_audio[i][j]))[0]
                    f.write(f"  .hword 0x{hex_val:04X}\n")
    
    if DUMP_TWIDDLE:
        PATH = os.path.join("DEMO/AudioPreProcess/DATA/PARAM", f"twiddle_factors.hpp")
        with open(PATH, "w") as f:
            f.write("#ifndef TWIDDLE_FACTORS_HPP\n")
            f.write("#define TWIDDLE_FACTORS_HPP\n")
            f.write("#include\"IntComplex.hpp\"\n")
            f.write("using namespace std;\n")

    # 3. Apply a Hann window to each frame
    window = window.mul(2**8).round().clamp(0,2**8-1).to(torch.int32)
    windowed_audio = (unfolded_audio * window.view([1]*(unfolded_audio.dim()-1)+[-1])).bitwise_right_shift(8)

    if WINDOW_GOLDEN:
        for i in range(NUM_OF_FRAMES):
            FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
            PATH = os.path.join(FRAME_BASE, 'windowed_audio.hex')
            with open(PATH, "w") as f:
                for j in range(512):
                    MSP, LSP = struct.unpack(">BB", struct.pack(">h", windowed_audio[i][j])) # big endian
                    f.write(f"{MSP:02X} {LSP:02X}\n")
            PATH = os.path.join(FRAME_BASE, 'window_audio.dec')
            with open(PATH, "w") as f:
                for j in range(512):
                    f.write(f"{windowed_audio[i][j]}\n")
            
                     
    windowed_audio = torch.cat([windowed_audio.unsqueeze(-1), torch.zeros_like(windowed_audio.unsqueeze(-1))], dim=-1) # Add complex dimension


    if SHOW_ERROR:
        orig_window = torch.hann_window(N_FFT).to(audio.device)
        orig_windowed_audio = audio.unfold(dimension=-1, size=N_FFT, step=HOP_LENGTH) * orig_window.view([1]*(unfolded_audio.dim()-1)+[-1])
        print("Windowed Audio Quantize Abs Error less than 1e-4: ", torch.allclose(windowed_audio[...,0].to(orig_windowed_audio.dtype).div(2**DECIMAL_BITS), orig_windowed_audio, atol=1e-4))
        print("Windowed Audio Quantize Abs Error less than 1e-3: ", torch.allclose(windowed_audio[...,0].to(orig_windowed_audio.dtype).div(2**DECIMAL_BITS), orig_windowed_audio, atol=1e-3))
        print("Windowed Audio Quantize Abs Error less than 1e-2: ", torch.allclose(windowed_audio[...,0].to(orig_windowed_audio.dtype).div(2**DECIMAL_BITS), orig_windowed_audio, atol=1e-2))
        print("Windowed Audio Quantize Abs Error less than 1e-1: ", torch.allclose(windowed_audio[...,0].to(orig_windowed_audio.dtype).div(2**DECIMAL_BITS), orig_windowed_audio, atol=1e-1))
        

    # 4. Compute the FFT of each frame
    stft_fft = int_fft(windowed_audio)[:,:N_FFT//2+1,:]  # Transpose to (batch, n_frames, n_bins, 2) for complex numbers
    if DUMP_TWIDDLE:
        PATH = os.path.join("DEMO/AudioPreProcess/DATA/PARAM", f"twiddle_factors.hpp")
        with open(PATH, "a") as f:
            f.write("#endif // TWIDDLE_FACTORS_HPP")

    if SHOW_ERROR:
        torch_fft = torch.fft.fft(orig_windowed_audio)[:,:N_FFT//2+1]
        print("INT FFT & TorchDFT Abs Error less than 1e-4: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), torch_fft, atol=1e-4))
        print("INT FFT & TorchDFT Abs Error less than 1e-3: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), torch_fft, atol=1e-3))
        print("INT FFT & TorchDFT Abs Error less than 1e-2: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), torch_fft, atol=1e-2))
        print("INT FFT & TorchDFT Abs Error less than 1e-1: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), torch_fft, atol=1e-1))
        float_fft = fft(orig_windowed_audio)[:,:N_FFT//2+1]
        print("FFT & TorchFFT Abs Error less than 1e-4: ", torch.allclose(torch_fft, float_fft, atol=1e-4))
        print("FFT & TorchFFT Abs Error less than 1e-3: ", torch.allclose(torch_fft, float_fft, atol=1e-3))
        print("FFT & TorchFFT Abs Error less than 1e-2: ", torch.allclose(torch_fft, float_fft, atol=1e-2))
        print("FFT & TorchFFT Abs Error less than 1e-1: ", torch.allclose(torch_fft, float_fft, atol=1e-1))
        
        print("INT FFT & Float FFT Abs Error less than 1e-4: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), float_fft, atol=1e-4))
        print("INT FFT & Float FFT Abs Error less than 1e-3: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), float_fft, atol=1e-3))
        print("INT FFT & Float FFT Abs Error less than 1e-2: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), float_fft, atol=1e-2))
        print("INT FFT & Float FFT Abs Error less than 1e-1: ", torch.allclose(torch.view_as_complex(stft_fft.float().div(2**DECIMAL_BITS)), float_fft, atol=1e-1))

    if FFT_GOLDEN:
        for i in range(NUM_OF_FRAMES):
            FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
            PATH = os.path.join(FRAME_BASE, 'fft_result.hex')
            with open(PATH, "w") as f:
                f.write("REAL|IMAG\n")
                for j in range(257):
                    MSP, LSP = struct.unpack(">BB", struct.pack(">h", stft_fft[i][j][0])) # big endian
                    f.write(f"{MSP:02X}{LSP:02X}|")
                    MSP, LSP = struct.unpack(">BB", struct.pack(">h", stft_fft[i][j][1])) # big endian
                    f.write(f"{MSP:02X}{LSP:02X}\n")
            PATH = os.path.join(FRAME_BASE, 'fft_result.dec')
            with open(PATH, "w") as f:
                f.write("  REAL|IMAG\n")
                for j in range(257):
                    f.write(f"{stft_fft[i][j][0]:06}|{stft_fft[i][j][1]:06}\n")

    magnitudes = stft_fft[...,0] ** 2 + stft_fft[...,1] ** 2  # calculate magnitude from real and imaginary parts
    magnitudes = magnitudes.t()[..., :-1].bitwise_right_shift(DECIMAL_BITS) # Drop last frame
    for i in range(NUM_OF_FRAMES):
        FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
        if not os.path.exists(FRAME_BASE):
            os.makedirs(FRAME_BASE)
        if SPECTRUM_GOLDEN:
            PATH = os.path.join(FRAME_BASE, 'spectrum.hex')
            with open(PATH, "w") as f:
                for j in range(257):
                    MSP, MSP2, LSP2, LSP = struct.unpack(">BBBB", struct.pack(">i", magnitudes[j][i])) # big endian
                    f.write(f"{MSP:02X}{MSP2:02X}{LSP2:02X}{LSP:02X}\n")
            PATH = os.path.join(FRAME_BASE, 'spectrum.dec')
            with open(PATH, "w") as f:
                for j in range(257):
                    f.write(f"{magnitudes[j][i]}\n")

    if SHOW_ERROR:
        # Check if the magnitude is close to the original STFT
        float_magnitudes = stft_origin[..., :-1].abs() ** 2
        print("STFT Magnitude & TorchDFT Magnitude Abs Error less than 1e-2: ", torch.allclose(magnitudes.div(2**DECIMAL_BITS), float_magnitudes, atol=1e-2))
        print("STFT Magnitude & TorchDFT Magnitude Abs Error less than 1e-1: ", torch.allclose(magnitudes.div(2**DECIMAL_BITS), float_magnitudes, atol=1e-1))
        print("STFT Magnitude & TorchDFT Magnitude Abs Error less than 1: ", torch.allclose(magnitudes.div(2**DECIMAL_BITS), float_magnitudes, atol=1))
        print("STFT Magnitude & TorchDFT Magnitude Abs Error less than 1e1: ", torch.allclose(magnitudes.div(2**DECIMAL_BITS), float_magnitudes, atol=1e1))

    filters = get_mel_filters(audio.device, n_mels=n_mels, n_fft=N_FFT)
    filters_shmnt = torch.log2(filters.max()).round().abs().int() + 7 # Scale to fit in int32
    filters = filters.mul(2**filters_shmnt).round().clamp(0,2**7).int()  # Quantize to int32
    mel_spec = (filters @ magnitudes).bitwise_right_shift(filters_shmnt) # Project to Mel spectrogram

    if DUMP_FILTER:
        PATH = os.path.join("DEMO/AudioPreProcess/DATA/PARAM", f"filters.hpp")
        with open(PATH, "w") as f:
            f.write("#ifndef FILTERS_HPP\n")
            f.write("#define FILTERS_HPP\n")
            f.write("#include<stdint.h>\n")
            f.write(f"volatile const uint8_t window_filter[{N_FFT}]"+" __attribute__((section(\".data.windowfilter\"))) = {\n")
            for i in range(window.shape[0]):
                f.write(f"{window[i]}")
                if i != window.shape[0]-1:
                    f.write(", ")
                else:
                    f.write("\n")
            f.write("};\n")

            f.write(f"volatile const uint8_t mel_filters[{n_mels*(N_FFT//2+1)}]"+" __attribute__((section(\".data.melfilters\"))) = {\n")
            filters = filters.flatten()
            for i in range(filters.shape[0]):
                f.write(f"{filters[i]}")
                if i != filters.shape[0]-1:
                    f.write(", ")
                else:
                    f.write("\n")
            f.write("};\n")
            f.write("#endif // FILTERS_HPP")
            
    if MEL_GOLDEN:
        for i in range(NUM_OF_FRAMES):
            FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
            if not os.path.exists(FRAME_BASE):
                os.makedirs(FRAME_BASE)
            PATH = os.path.join(FRAME_BASE, 'mel_result.hex')
            with open(PATH, "w") as f:
                for j in range(mel_spec.shape[0]):
                    MSP, LSP = struct.unpack(">BB", struct.pack(">H", mel_spec[j][i])) # big endian
                    f.write(f"{MSP:02X}{LSP:02X}\n")
            PATH = os.path.join(FRAME_BASE, 'mel_result.dec')
            with open(PATH, "w") as f:
                for j in range(mel_spec.shape[0]):
                    f.write(f"{mel_spec[j][i]}\n") 
    if SHOW_ERROR:
        # Check if the Mel spectrogram is close to the original STFT
        float_filters = get_mel_filters(audio.device, n_mels=n_mels, n_fft=N_FFT)
        float_mel_spec = float_filters @ float_magnitudes
        print("Mel Spectrogram & TorchDFT Mel Spectrogram Abs Error less than 1e-2: ", torch.allclose(mel_spec, float_mel_spec, atol=1e-2))
        print("Mel Spectrogram & TorchDFT Mel Spectrogram Abs Error less than 1e-1: ", torch.allclose(mel_spec, float_mel_spec, atol=1e-1))
        print("Mel Spectrogram & TorchDFT Mel Spectrogram Abs Error less than 1: ", torch.allclose(mel_spec, float_mel_spec, atol=1))
        print("Mel Spectrogram & TorchDFT Mel Spectrogram Abs Error less than 1e1: ", torch.allclose(mel_spec, float_mel_spec, atol=1e1))

    if result_alpha is not None:
        table = {}
        table_index = []
        output_table = torch.arange(-128, 128, dtype=torch.float32, device=audio.device)
        apart_table = torch.arange(-127.5, 127.5, 1.0, dtype=torch.float32, device=audio.device) / result_alpha
        apart_table = (apart_table-1)*4
        apart_table = 10**apart_table
        apart_table = apart_table.mul(2**DECIMAL_BITS).round().to(torch.int32).clamp(0,2**16)  # Quantize to int8
        for i in reversed(range(apart_table.shape[0])):
            if not apart_table[i].item() in table:
                table[apart_table[i].item()] = output_table[i+1].item()
                table_index.append(apart_table[i].item())

        output = torch.ones_like(mel_spec, dtype=torch.float32, device=audio.device) * (0) * result_alpha

        for i in reversed(range(len(table_index))):
            output[mel_spec > table_index[i]] = table[table_index[i]]
        
        if DUMP_LOGLUT:
            PATH = os.path.join("DEMO/AudioPreProcess/DATA/PARAM", f"lutlog.hpp")
            with open(PATH, "w") as f:
                f.write("#ifndef LUTLOG_HPP\n")
                f.write("#define LUTLOG_HPP\n")
                f.write("#include<stdint.h>\n")
                f.write(f"const int lut_size = {len(table_index)};\n")
                f.write(f"const int32_t lut_index[{len(table_index)}]"+" __attribute__((section(\".data.lutindex\"))) = {\n")
                for i in reversed(range(len(table_index))):
                    f.write(f"    {table_index[i]}")
                    if i != 0:
                        f.write(",\n")
                    else:
                        f.write("\n")
                f.write("};\n")

                f.write(f"const int8_t lut_value[{len(table_index)}]"+" __attribute__((section(\".data.melfilters\"))) = {\n")
                for i in reversed(range(len(table_index))):
                    f.write(f"    {int(table[table_index[i]])}")
                    if i != 0:
                        f.write(",\n")
                    else:
                        f.write("\n")
                f.write("};\n")
                f.write("#endif // LUTLOG_HPP\n")
        
        if LOGMEL_GOLDEN:
            for i in range(NUM_OF_FRAMES):
                FRAME_BASE = os.path.join(GOLDEN_BASE, f'frame_{i:04d}')
                if not os.path.exists(FRAME_BASE):
                    os.makedirs(FRAME_BASE)
                PATH = os.path.join(FRAME_BASE, 'logmel_result.hex')
                with open(PATH, "w") as f:
                    for j in range(output.shape[0]):
                        f.write(f"{output[j, i].int():02X}\n")

                PATH = os.path.join(FRAME_BASE, 'logmel_result.dec')
                with open(PATH, "w") as f:
                    for j in range(output.shape[0]):
                        f.write(f"{output[j, i].int()}\n")

        if int_mode:
            return output.to(torch.int8)
        else:
            return output / result_alpha
    else:
        mel_spec = mel_spec.float().div(2**DECIMAL_BITS)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = log_spec/4.0 + 1.0
        # print('min:' ,log_spec.min(),'max:', log_spec.max())
        return log_spec
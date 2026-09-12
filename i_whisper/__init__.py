import hashlib
import io
import os
from os import path
import urllib
import warnings
import ctypes, gc
from typing import Callable, List, Optional, Union

import torch
from tqdm import tqdm

from .audio import load_audio, log_mel_spectrogram, pad_or_trim, log_mel_spectrogram, log_mel_spectrogram_fft
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import ModelDimensions, Whisper
from .model_quant import Whisper_quantized
from .model_int import Whisper_int
from .transcribe import transcribe
from .version import __version__
from .gpu_mem_track import MemTracker

_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
}


def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as f:
            model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location='cpu')
    del checkpoint_file
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)

def load_model_quantized(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
    quantize: bool = False,
) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location='cpu')
    del checkpoint_file
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    dims = ModelDimensions(**checkpoint["dims"])
    model_o = Whisper(dims)
    model_o.load_state_dict(checkpoint["model_state_dict"])
    
    if quantize:
        model = Whisper_quantized(dims, quantize=True)
    else:
        model = Whisper_quantized(dims)
        result = model.load_state_dict(model_o.state_dict(), strict=False)
        print("From original model load ckpt to quantized model")
        print("Missing keys not in CKPT: ", len(result.missing_keys), ", showing up to 10: ", result.missing_keys[:10])
        print("Unexpected keys in CKPT but not in Model: ", len(result.unexpected_keys), ", showing up to 10: ", result.unexpected_keys[:10])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)



import json
from dataclasses import asdict
from pathlib import Path
from safetensors.torch import load_file, save_file
_INT_CKPT_ROOT = Path("CKPT")
_INT_FORMAT = "i-whisper-int"
_INT_FORMAT_VERSION = 1



def _resolve_int_model_dir(name: str) -> Path:
    path = Path(name)

    # Explicit model directory
    if path.is_dir():
        return path

    # Explicit path to model.safetensors
    if path.is_file():
        if path.name != "model.safetensors":
            raise RuntimeError(
                f"Expected model.safetensors, got: {path}"
            )
        return path.parent

    # Model alias: "base" -> CKPT/iwhisper-base
    model_dir = _INT_CKPT_ROOT / f"iwhisper-{name}"

    if model_dir.is_dir():
        return model_dir

    raise RuntimeError(
        f"I-Whisper INT model '{name}' not found. "
        f"Expected directory: {model_dir}"
    )


def save_int_model(
    model: Whisper_int,
    prefix: str,
    model_name: str,
) -> None:
    """
    Save an I-Whisper integer-only model.

    The model is stored as:
        <prefix>/iwhisper-<model_name>/
            config.json
            model.safetensors

    Parameters
    ----------
    model : Whisper_int
        Integer-only Whisper model.
    prefix : str
        Root directory for model artifacts.
    model_name : str
        Model variant, e.g. "tiny", "base", "small".
    """
    if not isinstance(model, Whisper_int):
        raise TypeError("model must be an instance of Whisper_int")

    model_dir = Path(prefix) / f"iwhisper-{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "format": _INT_FORMAT,
        "format_version": _INT_FORMAT_VERSION,
        "model_name": model_name,
        "dims": asdict(model.dims),
    }

    # Safetensors requires dense contiguous tensors.
    # Always serialize the canonical model state on CPU.
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }

    save_file(
        state_dict,
        str(model_dir / "model.safetensors"),
        metadata={
            "format": _INT_FORMAT,
            "format_version": str(_INT_FORMAT_VERSION),
        },
    )

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Integer-only model saved successfully at {model_dir}")



def load_int_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
) -> Whisper_int:
    """
    Load an I-Whisper integer-only model.

    Parameters
    ----------
    name : str
        Either:
        - a model name, e.g. "base"
        - a model directory, e.g. "CKPT/iwhisper-base"
        - a path to model.safetensors
    device : Union[str, torch.device], optional
        Target device.

    Returns
    -------
    model : Whisper_int
        Loaded integer-only Whisper model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = _resolve_int_model_dir(name)

    config_path = model_dir / "config.json"
    tensor_path = model_dir / "model.safetensors"

    if not config_path.is_file():
        raise RuntimeError(
            f"Missing model configuration: {config_path}"
        )

    if not tensor_path.is_file():
        raise RuntimeError(
            f"Missing model tensors: {tensor_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("format") != _INT_FORMAT:
        raise RuntimeError(
            f"Unsupported model format: {config.get('format')}"
        )

    if config.get("format_version") != _INT_FORMAT_VERSION:
        raise RuntimeError(
            f"Unsupported model format version: "
            f"{config.get('format_version')}"
        )

    dims = ModelDimensions(**config["dims"])

    # Construct the canonical model first.
    model = Whisper_int(dims)

    # Safetensors preserves the original integer dtypes.
    state_dict = load_file(
        str(tensor_path),
        device="cpu",
    )

    # strict=True is important:
    # the serialized artifact must exactly match Whisper_int.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    del state_dict
    gc.collect()

    model = model.to(device)
    model.eval()

    return model


def _tensor_exact_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """
    Bit/value-exact tensor comparison while handling PyTorch unsigned
    integer dtypes whose torch.equal kernels may not be implemented.
    """
    if a.dtype in (torch.uint16, torch.uint32):
        return torch.equal(
            a.to(torch.int64),
            b.to(torch.int64),
        )

    return torch.equal(a, b)


def verify_int_model_roundtrip(
    original: Whisper_int,
    loaded: Whisper_int,
    streamer: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Verify that two Whisper INT models have identical persistent states
    after save/load round-trip.

    Parameters
    ----------
    original : Whisper_int
        Original model before serialization.
    loaded : Whisper_int
        Model reconstructed from the serialized checkpoint.
    streamer : Callable[[str], None], optional
        Output function for verification messages.
        Defaults to print.
    """
    if streamer is None:
        streamer = print

    original_state = original.state_dict()
    loaded_state = loaded.state_dict()

    if original_state.keys() != loaded_state.keys():
        missing = original_state.keys() - loaded_state.keys()
        unexpected = loaded_state.keys() - original_state.keys()

        if missing:
            streamer(f"Missing keys: {sorted(missing)}")

        if unexpected:
            streamer(f"Unexpected keys: {sorted(unexpected)}")

        raise AssertionError("State dict keys mismatch")

    for name in original_state:
        a = original_state[name].detach().cpu()
        b = loaded_state[name].detach().cpu()

        if a.dtype != b.dtype:
            raise AssertionError(
                f"{name}: dtype mismatch: {a.dtype} != {b.dtype}"
            )

        if a.shape != b.shape:
            raise AssertionError(
                f"{name}: shape mismatch: {a.shape} != {b.shape}"
            )

        if not _tensor_exact_equal(a, b):
            raise AssertionError(
                f"{name}: tensor value mismatch"
            )

    streamer(
        f"Integer model round-trip verification passed "
        f"({len(original_state)} tensors checked)"
    )
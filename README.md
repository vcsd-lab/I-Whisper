# I-Whisper: Integer-only Quantization for Whisper Inference and Deployment

This repository contains the official implementation of **I-Whisper**, a fully static integer-only quantization framework for OpenAI Whisper models.

I-Whisper reformulates key Whisper operators to enable hardware-friendly W8A8 inference without runtime floating-point rescaling, dynamic activation smoothing, or token-dependent quantization adaptation.

> **Paper:** [arXiv link coming soon]  
<!-- > **Conference:** APCCAS 2026 Special Session   -->
<!-- > **Code:** https://github.com/vcsd-lab/I-Whisper -->

---

## Overview

Whisper is a strong encoder--decoder Transformer for automatic speech recognition, but fully static integer-only deployment is challenging due to post-GeLU activation outliers, LayerNorm numerical instability, and length-sensitive Softmax normalization.

I-Whisper addresses these issues with three operator-level reformulations:

- **GeLU reparameterization** for suppressing post-nonlinearity FFN outliers.
- **Row-wise block-aligned integer LayerNorm** for stable low-precision normalization.
- **Attention softmax fusion** for avoiding direct quantization of normalized Softmax probabilities.

All quantization parameters are calibrated offline, and the inference path avoids runtime floating-point fallback.

---

## Method Summary

I-Whisper targets fully static W8A8 inference for Whisper.

| Component | Design |
|---|---|
| Linear weights | INT8, per-output-channel |
| Activations | INT8, primarily per-tensor |
| FFN nonlinearity | SmoothQuant-style GeLU reparameterization |
| LayerNorm | INT15 input path with row-wise aligned accumulation |
| Attention | Exponential-value accumulation with deferred normalization |
| Normalization | Fast inverse-square-root based integer backend |

---

## Results

I-Whisper is evaluated on LibriSpeech `test-other` across multiple Whisper model scales.

| Method | Tiny | Base | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| FP32 | 18.88 | 14.57 | 10.08 | 8.46 | 6.90 |
| Naive W8A8 | 55.92 | 37.03 | 33.42 | 24.00 | 84.38 |
| I-Whisper | 24.53 | 19.17 | 14.95 | 10.07 | 13.58 |

WER is reported on successfully decoded samples. Lower is better.

---

## Installation

```sh
git clone https://github.com/vcsd-lab/I-Whisper.git Whisper
cd I-Whisper
# Install requirements
pip install -f requirements.txt
# put LibriSpeech into project root
cp -r your-LibriSpeech-Path .
```

## Usage
### Put the audio data want to transcribe
```sh
mv data.wav vcsdl_dataset/.
```
### RUN Whisper quant routine & Fake quantize version
```sh
python3 quantize.py [-h] [--model MODEL] [--calibrate] [--smooth] [--device DEVICE] [--test] [--sample SAMPLE] [--logdir LOGDIR] [--device] [--filename] [--debug]  
# PyTorch Whisper quantize and test routine

# options:
#   -h, --help       show this help message and exit
#   --model MODEL    model type (size of model: tiny, base, small, medium, large), defualt base
#   --calibrate      set to calibrate quantzation params when quantize start, default False
#   --smooth         set to smooth params beform quantization, defualt False
#   --device DEVICE  device to use for computation (cuda, cpu), default cuda
#   --test           use LibriSpeech/test-other dataset to test or not, if False then transcribe audio data FILENAME
#   --sample SAMPLE  number of samples in dataset to test, defualt all (recommend 10)
#   --logdir LOGDIR  directory to save logs
#   --device DEVICE  device to run, cuda or cpu
#   --filename FILENAME transcribe FILENAME in vcsdl_dataset
#   --debug

# for example, calibrate and smooth with LibriSpeech and transcribe file0.wav in vcsdl_dataset on cpu, using model size base
python3 quantize.py --calibrate --smooth --model base --device cuda --filename file0.wav
```

## Citation

If you find I-Whisper useful, please cite our paper:
```tex
@misc{chen2026iwhisper,
  title        = {I-Whisper: Integer-only Quantization for Whisper Inference and Deployment},
  author       = {Jun-An Chen and Shih-Ping Chang and Chih-Hung Kuo},
  year         = {2026},
  eprint       = {ARXIV_ID_PLACEHOLDER},
  archivePrefix= {arXiv},
  primaryClass = {cs.SD},
  url          = {ARXIV_URL_PLACEHOLDER}
}
```




## License

This project is released under the MIT License.

This repository contains code derived from or adapted from
[OpenAI Whisper](https://github.com/openai/whisper), which is also released
under the MIT License. We retain the original OpenAI copyright and license
notice where applicable.

Modifications and additional code for I-Whisper are copyright of
Video and Communication System Design Lab, National Cheng Kung University.


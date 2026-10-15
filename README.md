# I-Whisper: Integer-Only Quantization for Whisper Inference and Deployment

**I-Whisper** is a research implementation of a statically quantized, integer-only neural-network inference path for [OpenAI Whisper](https://github.com/openai/whisper), developed at the Video and Communication System Design Lab, National Cheng Kung University.

I-Whisper focuses on hardware-friendly integer computation in Whisper's **Audio Encoder and Text Decoder**, with implementations and evaluation on CPU platforms including the Raspberry Pi 5.

This repository accompanies our research presented at **APCCAS 2026**.

## Overview

Whisper is an encoder–decoder Transformer for automatic speech recognition (ASR). Deploying Whisper with static integer arithmetic introduces numerical challenges in nonlinear functions, normalization, and attention.

I-Whisper addresses these challenges through three principal techniques:

- **GeLU reparameterization:** Mitigates activation outliers in feed-forward networks.
- **Integer LayerNorm:** Uses row-wise block-aligned arithmetic to support fixed-width integer accumulation and normalization.
- **Attention softmax reformulation:** Combines exponential-value accumulation and deferred normalization to reduce reliance on floating-point softmax computation.

The implementation primarily uses INT8 weights and activations, with wider integer representations for intermediate arithmetic as required.

### Scope of integer-only inference

The term *integer-only* refers specifically to the quantized **Whisper Encoder and Decoder neural-network computation**.

The reference transcription workflow retains the standard Whisper floating-point audio frontend, including waveform normalization and Log-Mel spectrogram computation. The final decoder output is also dequantized when required by the standard token-decoding procedure.

Consequently, I-Whisper does **not** claim that the entire end-to-end ASR application is free of floating-point operations.

Experimental integer audio-preprocessing implementations are present in the research codebase but are not part of the validated inference path described here.

## Method Summary

| Component | Implementation |
|---|---|
| Linear weights | INT8, per-output-channel quantization |
| Activations | Primarily per-tensor INT8 quantization |
| Feed-forward network | GeLU reparameterization and offline calibration |
| Residual and normalization paths | Wider fixed-point integer representations |
| LayerNorm | Row-wise block-aligned integer computation |
| Attention | Integer-oriented exponential accumulation and deferred normalization |
| Quantization parameters | Determined offline |

The default SIMD LayerNorm implementation uses an INT32-oriented accumulation strategy. An alternative INT64 implementation is retained for research comparisons and numerical analysis; it is not the default inference configuration.

## Evaluation Results

I-Whisper was evaluated on the **LibriSpeech `test-other`** dataset using multiple Whisper model sizes.

Word Error Rate (WER, %, lower is better):

| Method | Tiny | Base | Small | Medium | Large |
|---|---:|---:|---:|---:|---:|
| FP32 | 18.88 | 14.57 | 10.08 | 8.46 | 6.90 |
| Naive W8A8 | 55.92 | 37.03 | 33.42 | 24.00 | 84.38 |
| I-Whisper (filtered) | 25.23 | 19.36 | 13.98 | 10.89 | 12.98 |
| I-Whisper (all-sample) | 26.86 | 20.58 | 30.99 | 11.34 | 18.33 |

The filtered results evaluate successfully decoded samples, whereas the all-sample results use the full evaluation set according to the corresponding experiment protocol. These results use different evaluation denominators and should not be treated as interchangeable.

The published research results above are distinct from the small-sample functional smoke tests described later in this README.

## Repository Structure

```text
I-Whisper/
├── i_whisper/                   # Whisper models, quantization and kernels
│   └── kernels/
│       └── csrc/                # C++ scalar and SIMD implementations
├── tests/                       # Kernel correctness tests
├── analysis/                    # Research and numerical analysis utilities
├── quantize.py                  # Quantization and calibration workflow
├── int_transcribe.py            # Integer model transcription/evaluation
├── check_integer_model.py       # Integer-only model-core audit
├── requirements.txt             # Common Python dependencies
├── requirements-rpi.txt         # Raspberry Pi 5 validated environment
├── constraints-ngc24.txt        # NVIDIA NGC 24.05 dependency constraints
└── setup.py                     # Native extension build configuration
```

## Installation

### Workstation: NVIDIA NGC PyTorch 24.05

The validated workstation setup uses the NVIDIA container:

```text
nvcr.io/nvidia/pytorch:24.05-py3
```

The container provides the NVIDIA PyTorch and CUDA environment. A platform-specific constraints file is used to preserve compatible dependencies, including the container-provided PyTorch build.

First, clone the repository:

```bash
git clone https://github.com/vcsd-lab/I-Whisper.git
cd I-Whisper
git submodule update --init i_whisper/kernels/csrc/simde
```

Start the NVIDIA container from the repository root:

```bash
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v "$PWD":/workspace/I-Whisper \
    -w /workspace/I-Whisper \
    nvcr.io/nvidia/pytorch:24.05-py3 \
    bash
```

Inside the container, install FFmpeg:

```bash
apt-get update
apt-get install -y --no-install-recommends ffmpeg
```

Install Python dependencies using the validated NGC 24.05 constraints:

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
    PIP_CONFIG_FILE=/dev/null \
    python3 -m pip install \
    --index-url https://pypi.org/simple \
    -c constraints-ngc24.txt \
    -r requirements.txt
```

Build the I-Whisper native extension:

```bash
python3 setup.py build_ext --inplace --force
```

Run the kernel test suite:

```bash
python3 -m pytest -q tests
```

**Validated result:** 1,499 tests passed on the NVIDIA NGC 24.05 workstation environment.

### Raspberry Pi 5

I-Whisper has also been validated on a Raspberry Pi 5 using native ARM NEON and DOTPROD implementations.

The tested environment uses Python 3.13.5 and a CPU-specific PyTorch 2.14.0 build. The recorded Python dependencies are available in `requirements-rpi.txt`.

The Raspberry Pi setup requires a compatible, separately installed CPU PyTorch build. The requirements snapshot alone does not guarantee availability of that exact wheel from the default Python package index.

After preparing the validated Python environment:

```bash
git submodule update --init i_whisper/kernels/csrc/simde
python3 setup.py build_ext --inplace --force
python3 -m pytest -q tests
```

**Validated result:** 1,499 kernel tests passed on Raspberry Pi 5.

The native ARM kernels and the x86 SIMDe compatibility implementation are distinct platform builds and should not be assumed to have identical performance.

## Integer Model Checkpoints

Serialized integer models use the following directory format:

```text
CKPT/
└── iwhisper-base/
    ├── config.json
    └── model.safetensors
```

The `config.json` file records the model dimensions and checkpoint format version. The `model.safetensors` file stores the model's persistent tensor state.

These artifacts can be loaded using:

```python
import i_whisper

model = i_whisper.load_int_model("base", device="cpu")
```

The loader accepts a model alias, an explicit model directory, or a path to `model.safetensors`.

Model checkpoints and LibriSpeech data are not included in the Git repository. Users must prepare compatible checkpoints before running the serialized integer-model transcription tools.

### Generating a quantized checkpoint

The research quantization workflow is implemented in `quantize.py`. It requires the original Whisper model and calibration data.

The script supports offline smoothing, activation calibration, integer-model conversion, and serialized checkpoint export. Checkpoint export through `--save_model` is performed in the script's evaluation path.

Calibration settings and datasets affect the resulting model accuracy. Reproducing the reported research WER requires the corresponding experimental configuration; merely generating a checkpoint is not a guarantee of obtaining the published results.

## Running Integer Inference

### End-to-End Quantization and Inference Smoke Test

The following command loads the original Whisper Base checkpoint, performs offline smoothing and quantization calibration, evaluates ten samples from the local LibriSpeech `test-other` dataset, and exports the integer model:

```bash
python3 quantize.py \
    --model base \
    --calibrate \
    --smooth \
    --test \
    --sample 10 \
    --save_model
```

The generated checkpoint is saved to:

```text
CKPT/iwhisper-base/
├── config.json
└── model.safetensors
```

The export procedure also verifies the saved and reloaded model states for exact tensor equality.

To evaluate the newly serialized integer model using the CPU backend:

```bash
python3 int_transcribe.py \
    --model base \
    --device cpu \
    --test \
    --sample 1
```

This workflow was successfully exercised on the validated NVIDIA NGC PyTorch 24.05 environment.

**Note:** The reference quantization script currently uses LibriSpeech `test-other` for calibration as well as evaluation. This example is intended to verify functional correctness and checkpoint reproducibility, not to establish an independent held-out WER benchmark.

### Single-audio transcription

For the current reference script, place an audio file at:

```text
vcsdl_dataset/tt.mp3
```

Then run:

```bash
python3 int_transcribe.py --model base --device cpu
```

The current `int_transcribe.py` single-file mode uses this fixed input path. It is not yet configurable through a filename command-line argument.

The script loads the integer checkpoint, performs the standard Whisper audio frontend, and executes the quantized model using the SIMD backends.

### LibriSpeech evaluation

Prepare a local copy of LibriSpeech `test-other` accessible in the container or host environment.

The current research evaluation script expects the local dataset path:

```text
LibriSpeech.test-other/test_other
```

It also accesses the original `.trans.txt` transcription files through paths derived from the dataset's audio filenames. The original directory structure must therefore be retained.

To perform a small functional evaluation:

```bash
python3 int_transcribe.py \
    --model base \
    --device cpu \
    --test \
    --sample 10
```

Remove `--sample` to evaluate the available dataset instead of a small subset.

**Warning:** Evaluation can overwrite the existing model report under `MODEL_TEST_REPOERT/`. Back up any existing results before rerunning this command.

A ten-sample WER is a functional smoke test, not a substitute for the full LibriSpeech evaluation reported above.

## Integer-Only Model Audit

A standalone audit utility is available to inspect the integer Encoder/Decoder forward path:

```bash
python3 check_integer_model.py --model base
```

By default, it uses a deterministic synthetic INT8 Mel input. An optional real audio file can be supplied:

```bash
python3 check_integer_model.py \
    --model base \
    --audio path/to/audio.wav
```

The audit temporarily disables final decoder dequantization and checks PyTorch-dispatched tensor operations for floating-point inputs and outputs.

**Audit scope:** Encoder and Decoder neural-network forward computation. Audio preprocessing, input quantization, output token decoding, and the internal arithmetic of native C++ kernels are outside this dispatch-level audit.

A successful result is reported as:

```text
Integer-only core audit: PASS
```

## Additional Research Utilities

The repository retains numerical-analysis tools under `analysis/`.

For example, the optional LayerNorm overflow diagnostic can be enabled with:

```bash
python3 int_transcribe.py \
    --model base \
    --device cpu \
    --debug
```

This records hypothetical overflow exposure for an unshifted integer accumulation formulation. It does not imply that the production LayerNorm kernel overflowed.

Diagnostic collection introduces runtime overhead, so timings obtained with `--debug` should not be compared directly with ordinary inference timings.

The alternative INT64 LayerNorm implementation is retained for numerical comparisons; INT32 SIMD remains the default inference backend.

## Reproducibility Notes

The tested workstation and Raspberry Pi environments intentionally use different PyTorch distributions and Python package versions.

The recommended approach is to use the validated environment for the target platform instead of automatically upgrading all Python dependencies.

Full WER reproducibility additionally depends on the model checkpoint, calibration data, dataset version, evaluation protocol, and decoding configuration.

## Authors and Acknowledgments

**Research and Implementation:** Jun-An Chen

**Affiliation:** Video and Communication System Design Lab,
Department of Electrical Engineering,
National Cheng Kung University, Taiwan.

I-Whisper was primarily researched and developed by Jun-An Chen
at National Cheng Kung University, based on OpenAI Whisper.

We acknowledge the original OpenAI Whisper authors and contributors,
as well as the developers of the open-source libraries used in this project.

## Publication

**Paper:** I-Whisper: Integer-Only Quantization for Whisper Inference and Deployment

**Status:** Accepted for oral presentation at the 2026 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS 2026).

The official IEEE Xplore record, DOI, and BibTeX citation will be added once the conference proceedings are published.

## License and Acknowledgments

I-Whisper is released under the [MIT License](LICENSE).

This repository incorporates code derived from or adapted from [OpenAI Whisper](https://github.com/openai/whisper), which is also distributed under the MIT License. The original OpenAI copyright and license notice are retained.

The native x86 compatibility build uses [SIMDe](https://github.com/simd-everywhere/simde), distributed as a Git submodule under its own MIT license and copyright notices.

We acknowledge the authors and contributors of OpenAI Whisper and the open-source libraries used in this research.

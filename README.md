# I-Whisper
This project focuses on quantization and deployment of Whisper models.
This repository includes code derived from OpenAI Whisper (MIT License).
## Install requirements
```sh
pip install -f requirements.txt
```

## Put the audio data want to transcribe
```sh
mv data.wav vcsdl_dataset/.
```

## RUN Whisper quant routine & Fake quantize version
```sh
python3 quantize.py [-h] [--model MODEL] [--device DEVICE] [--test] [--sample SAMPLE] [--logdir LOGDIR] [--device] [--filename] [--debug]  

# PyTorch Whisper quantize and test routine

# options:
#   -h, --help       show this help message and exit
#   --model MODEL    model type (size of model: tiny, base, large), defualt base
#   --device DEVICE  device to use for computation (cuda, cpu), default cuda
#   --test           use dataset to test or not
#   --sample SAMPLE  number of samples in dataset to test, defualt all (recommend 10)
#   --logdir LOGDIR  directory to save logs
#   --device DEVICE  device to run, cuda or cpu
#   --filename FILENAME transcribe FILENAME in vcsdl_dataset
#   --debug

# for example, transcribe file0.wav in vcsdl_dataset on cpu, using model size base
python3 quantize.py --model base --device cpu --filename file0.wav
```

import torch
import i_whisper
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
import logging
from termcolor import colored
import time
import argparse
import sys
from memory_profiler import profile

import inspect
from i_whisper.model import *
from i_whisper import MemTracker

import random
import jiwer
from i_whisper.model import ResidualAttentionBlock
from i_whisper.model_quant import ResidualAttentionBlock_quantized, LayerNorm_quantized, Linear_quantized, GELU_quantized

import re
class Expand(jiwer.AbstractTransform):
    def process_string(self, s: str):
        # definitely a non exhaustive list

        # specific words
        s = re.sub(r"THERE'S", "THERE IS", s)
        s = re.sub(r"DON'T", "DO NOT", s)
        s = re.sub(r"MR", "MISTER", s)
        s = re.sub(r"SHOPBOY", "SHOP BOY", s)
        s = re.sub(r"WORKHOUSE", "WORK HOUSE", s)
        return s


def String_Process(TEXT_LIST: list[str])-> list[str]:
        TEXT_LIST = jiwer.RemoveMultipleSpaces()(TEXT_LIST)
        TEXT_LIST = jiwer.Strip()(TEXT_LIST)
        TEXT_LIST = jiwer.ExpandCommonEnglishContractions()(TEXT_LIST)
        TEXT_LIST = jiwer.ToUpperCase()(TEXT_LIST)
        TEXT_LIST = Expand()(TEXT_LIST)
        TEXT_LIST = jiwer.RemovePunctuation()(TEXT_LIST)
        return TEXT_LIST


def WER_Process(gt_list, pred_list)->tuple:
    normal_gt = []
    normal_pred = []
    outlier_gt = []
    outlier_pred = []
    for i in range(len(gt_list)):
        score = jiwer.wer(gt_list[i], pred_list[i])
        if score < 1.0:
            normal_gt.append(gt_list[i])
            normal_pred.append(pred_list[i])
        else:
            outlier_gt.append(gt_list[i])
            outlier_pred.append(pred_list[i])
    return (normal_gt, normal_pred), (outlier_gt, outlier_pred)


def Align_and_Report(gt_list, pred_list, audio_length_list, file_handle):
    # String Processing of GT
    TEXT_GT = String_Process(gt_list)
    # String Processing of model output
    TEXT = String_Process(pred_list)

    NORMAL_TEXT, OUTLIER_TEXT = WER_Process(TEXT_GT, TEXT)
    REPORT_ALL = jiwer.process_words(TEXT_GT,TEXT)
    REPORT_NORMAL = jiwer.process_words(NORMAL_TEXT[0],NORMAL_TEXT[1])
    REPORT_OUTLIER = jiwer.process_words(OUTLIER_TEXT[0],OUTLIER_TEXT[1])

    print("=v="*40, file=file_handle)
    print("Overview of Sentences:", file=file_handle)
    print("> Probability of Extreme Outliers Sentences (WER>100%): {:.2f}%".format(100*len(OUTLIER_TEXT[0])/len(TEXT_GT)), file=file_handle)
    print("> Number of Extreme Outliers Sentences (WER>100%): {:4>}".format(len(OUTLIER_TEXT[0])), file=file_handle)
    print("> Number of Sentences: {:>4}".format(len(TEXT)), file=file_handle)
    print("Audio Length:", file=file_handle)
    for i in range(len(audio_length_list)):
        print('audio '+str(i)+' time is: '+str(audio_lengths[i]), file=file_handle)
    print("=^="*40, file=file_handle)
    print("", file=file_handle)

    print("=v="*40, file=file_handle)
    print("Detailed WER Alignment Results:", file=file_handle)
    print(jiwer.visualize_alignment(REPORT_ALL, skip_correct=False), file=file_handle)
    print("=^="*40, file=file_handle)
    print("", file=file_handle)

    print("=v="*40, file=file_handle)
    print('Detailed WER Alignment Results - Normal Cases (WER<=100%):\n', file=file_handle)
    print(jiwer.visualize_alignment(REPORT_NORMAL, skip_correct=False), file=file_handle)
    print("=^="*40, file=file_handle)
    print("", file=file_handle)
    
    print("=v="*40, file=file_handle)
    if len(OUTLIER_TEXT[0]) == 0:
        print("No Outlier Results (WER>100%)", file=file_handle)
    else:
        print('Detailed WER Alignment Results - Outlier Cases (WER>100%):\n', file=file_handle)
        print(jiwer.visualize_alignment(REPORT_OUTLIER, skip_correct=False), file=file_handle)
    print("=^="*40, file=file_handle)
    

class LoggerStream:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)
        # if message != '\n':
        #     self.logger.log(self.level, message)
    def flush(self):
        pass


if __name__ != "__main__":
    exit(0)
parser = argparse.ArgumentParser(description='PyTorch ViT Training')
parser.add_argument('--model', default='base', type=str, help='model type (size of model)')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device to use for computation')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--sample', default=None, type=int, help='number of samples to test')
parser.add_argument('--logdir', default='EXELOG', type=str, help='directory to save logs')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='increase output verbosity')
args = parser.parse_args()
# torch.set_printoptions(precision=8,sci_mode=False,threshold=500)

'''logging setup'''
log_dir = args.logdir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = logging.getLogger("main" if __name__ == "__main__" else __name__)

# console formatter
color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
            colored('(%(filename)s %(lineno)d)', 'yellow') + \
            ': |%(levelname)s| %(message)s' 
# create console handlers for master process
console_handler = logging.StreamHandler(sys.stdout)
if args.verbose:
    console_handler.setLevel(logging.DEBUG)
else:
    console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(console_handler)

# log file formatter
fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): |%(levelname)s| %(message)s'
# create file handlers
file_handler = logging.FileHandler(os.path.join(log_dir, f'{os.path.splitext(os.path.basename(__file__))[0]}_{int(time.time())}.log'), mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
logger.info(args.__str__())
# GPU memory tracker
gpu_tracker = MemTracker(detail=True, logging_fn=logger)
if torch.cuda.is_available():
    gpu_tracker.track()


'''Load integer preprocessing parameters'''
CKPT_PATH = os.path.join('CKPT', 'consolidated.int.'+args.model+'.pth')
profiler_stream = LoggerStream(logger, logging.INFO)
# @profile(stream=profiler_stream)
# def load_preproc():
#     PreProcessor = i_whisper.audio.PreProcessor(device="cpu", param_init=False)
#     PreProcessor.load_param()
#     return PreProcessor
# PreProcessor = load_preproc()

'''Load integer whisper model'''
if args.device == 'cuda' and torch.cuda.is_available():
    model = i_whisper.load_int_model(args.model, args.device)
    gpu_mem_log = gpu_tracker.track()
    for log_line in gpu_mem_log:
        logger.info(log_line)
else:
    @profile(stream=profiler_stream)
    def load():
        return i_whisper.load_int_model(args.model, args.device)
    model = load()
# Decoding options
options = i_whisper.DecodingOptions(fp16=False, language="en")

from i_whisper.modules.linear import set_linear8_backend, set_linear32_backend
set_linear8_backend(model, backend="simd", logger=logger)  # set backend for Linear8 layers
set_linear32_backend(model, backend="simd", logger=logger)  # set backend for Linear32 layers

from i_whisper.model_int import (
    set_residual_backend,
    set_layernorm_backend,
    set_gelu_backend,
    set_attention_backend,
    set_conv1d_backend,
    set_audio_encoder_backend,
    set_text_decoder_backend,
)
set_residual_backend(model, backend="simd", logger=logger)  # set backend for ResidualAttention
set_layernorm_backend(model, backend="simd", logger=logger)  # set backend for LayerNorm layers
set_gelu_backend(model, backend="simd", logger=logger)  # set backend for GELU layers
set_attention_backend(model, backend="simd", logger=logger)  # set backend for Attention layers
set_conv1d_backend(model, backend="simd", logger=logger)  # set backend for Conv1d layers
set_audio_encoder_backend(model, backend="simd", logger=logger)  # set backend for Add+Requant15 layers
set_text_decoder_backend(model, backend="simd", logger=logger)  # set backend for TextDecoder layers


if args.test:
    '''Load audio dataset & ground truth'''
    ds_name = "LibriSpeech.test-other/test_other"
    ds = load_dataset(ds_name)

    TEXT_GT = []
    TEXT = []
    
    '''Test results collection directory'''
    RESULT_HOME = os.path.join('MODEL_TEST_REPOERT')
    if not os.path.exists(RESULT_HOME):
        os.makedirs(RESULT_HOME)

    

    audio_lengths = []
    t1 = time.time()
    test_loop = tqdm( range(len(ds['train']) if args.sample is None else args.sample) )
    for i in test_loop:
        test_loop.set_description_str(f"Transcribe and Test with {ds_name}, Sample {i}")
        audio = ds['train'][i]["audio"]["array"]
        # audio = (np.float32(audio) * 2**15).astype(np.int16)  # to INT16 range

        audio_lengths.append(audio.shape[0]/16000)
        audio = i_whisper.pad_or_trim(audio)
        mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # Reading Ground Truth
        AUDIO_PATH = ds['train'][i]["audio"]["path"]
        GT_PATH = '-'.join(AUDIO_PATH.split('-')[:-1]) + '.trans.txt'
        GT_FILE = open(GT_PATH, 'r')
        GT_READ = GT_FILE.readlines()[int( AUDIO_PATH.split('-')[-1].split('.')[0] )].split(' ', 1)[1].strip('\n')
        # GT_PATH = ds['train'][i]["audio"]["path"]
        # GT_FILE = open(GT_PATH.split('-')[0] + '-' + GT_PATH.split('-')[1] +'.trans.txt', 'r')
        # GT_READ = GT_FILE.readlines()[int( GT_PATH.split('-')[-1].split('.')[0] )].split(' ', 1)[1].strip('\n')
        logger.info(f"AUDIO data path: {AUDIO_PATH}")
        # add GT to TEXT_GT list
        TEXT_GT.append(GT_READ)
        GT_FILE.close()

        # quantized model transcription
        result = i_whisper.decode(model, mel, options)
        logger.info(f"transcribe result:    {result.text}")
        logger.info(f"ground truth(answer): {GT_READ}")
        
        # add result to TEXT list
        TEXT.append(result.text)

    t2 = time.time()


    '''Logging and Reporting Integer Model Test Results'''
    RESULT_QUANT = os.path.join(RESULT_HOME, 'int_'+args.model+'.report')
    f = open(RESULT_QUANT, 'w') # clear previous contents
    f.close()
    f = open(RESULT_QUANT, 'a')
    print("="*50+" R E P O R T "+"="*50, file=f)
    print('Setups:', file=f)
    print('> Online-Quant Model (FP Scale)', file=f)
    print('> INT-FFT', file=f)
    print('> INT-MEL', file=f)
    print('> INT LOG', file=f)
    Align_and_Report(TEXT_GT, TEXT, audio_lengths, f)
    print(f"Total inference time for {len(TEXT)} samples: {t2-t1:.2f} seconds", file=f)
    f.close()

else:
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of Whisper {args.model} parameters: {total_params}")

    '''Load Audio & PreProcessing'''
    audio = i_whisper.load_audio("vcsdl_dataset/tt.mp3", raw=True)
    logger.info(f"Original audio length: {audio.shape[0]/16000:.2f} seconds")
    audio_list = []
    while audio.shape[0] > 30*16000:
        # cut the audio to many 30-second clips if too long
        audio_list.append( audio[:30*16000] )
        audio = audio[30*16000:]
    # the last clip (less than 30 seconds)
    audio = i_whisper.pad_or_trim(audio)
    audio_list.append(audio)
    del audio
    logger.info(f"Original Audio is cut into {len(audio_list)} 30-second clips")
    t1 = time.time()
    for idx, audio in enumerate(audio_list):
        # mel = PreProcessor.log_mel_spectrum(audio).to(model.device)
        mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        result = i_whisper.decode(model, mel, options)
        logger.info(f"clip {idx:>2d}: {result.text}")
    t2 = time.time()
    logger.info(f"Total inference time for {len(audio_list)} clips: {t2-t1:.2f} seconds")

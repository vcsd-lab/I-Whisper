import gc

import torch
import i_whisper
from datasets import load_dataset
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import logging
from termcolor import colored

import inspect
from i_whisper.model import *
from i_whisper import MemTracker
import argparse

import random
import jiwer
from i_whisper.model_quant import ResidualAttentionBlock_quantized, LayerNorm_quantized, Linear_quantized, GELU_quantized
from i_whisper.model_int import Whisper_int


import traceback
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten



class IntegerOnlyAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    @staticmethod
    def _has_float_tensor(obj):
        flat, _ = tree_flatten(obj)

        return any(
            isinstance(x, torch.Tensor)
            and (x.is_floating_point() or x.is_complex())
            for x in flat
        )

    @staticmethod
    def _describe_tensors(obj):
        flat, _ = tree_flatten(obj)

        ret = []

        for x in flat:
            if isinstance(x, torch.Tensor):
                ret.append(
                    f"shape={tuple(x.shape)}, dtype={x.dtype}"
                )

        return ret

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = kwargs or {}

        input_has_float = self._has_float_tensor(
            (args, kwargs)
        )

        out = func(*args, **kwargs)

        output_has_float = self._has_float_tensor(out)

        self.ops.append(str(func))

        if input_has_float or output_has_float:
            stack = "".join(
                # traceback.format_stack(limit=15)
                traceback.format_stack()
            )

            raise RuntimeError(
                "\n"
                "========================================\n"
                "[FLOAT TENSOR DETECTED]\n"
                f"Operator : {func}\n"
                f"Inputs   : {self._describe_tensors((args, kwargs))}\n"
                f"Outputs  : {self._describe_tensors(out)}\n"
                f"{stack}"
                "========================================"
            )

        return out



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
    


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, LayerNorm_quantized)
    for fc in fcs:
        assert isinstance(fc, Linear_quantized)
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
        fc.running_max_c.div_(scales)

@torch.no_grad()
def smooth_gelu_fcs(mlp0, gelu, mlp2, act_scales, alpha):
    assert isinstance(gelu, GELU_quantized)
    assert isinstance(mlp0, Linear_quantized)
    assert isinstance(mlp2, Linear_quantized)
    
    H = mlp0.out_features
    assert H == act_scales.numel()

    device, dtype = mlp2.weight.device, mlp2.weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = mlp2.weight.abs().amax(dim=0).clamp_min(1e-12)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    # quantize the scales to 255 levels (fake quantization)
    scales.div_(scales.max()/255.0).round_().mul_(scales.max()/255.0)
    mlp0.weight.div_(scales.view(-1, 1))
    if mlp0.bias is not None:
        mlp0.bias.div_(scales)
    gelu.de_smooth_vec.mul_(scales)
    mlp2.weight.mul_(scales.view(1, -1))
    mlp2.running_max_c.div_(scales)



if __name__ != "__main__":
    exit(0)
parser = argparse.ArgumentParser(description='I-Whisper quantization program')
parser.add_argument('--model', default='base', type=str, help='model type (size of model)')
parser.add_argument('--fakequant', default=False, action='store_true', help='whether to test fake quantized model')
parser.add_argument('--baseline', default=False, action='store_true')
parser.add_argument('--calibrate', default=False, action='store_true')
parser.add_argument('--smooth', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--sample', default=None, type=int, help='number of samples to test')
parser.add_argument('--save_model', default=False, action='store_true',)
parser.add_argument('--logdir', default='EXELOG', type=str, help='directory to save logs')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device to use for computation')
parser.add_argument('--filename', default='tt.mp3', type=str, help='filename for audio data to test')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
# torch.set_printoptions(precision=8,sci_mode=False,threshold=500)


ln_fcs_alpha = {
    'tiny': 0.4,
    'base': 0.7,
    'base.en': 0.5, # TODO: Re-Calibrate a good param
    'small': 0.0,
    'small.en': 0.5, # TODO: Re-Calibrate a good param
    'medium': 0.7,
    'large': 0.2,
}

gelu_fcs_alpha = {
    'tiny': 0.5,
    'base': 0.3,
    'base.en': 0.3, # TODO: Re-Calibrate a good param
    'small': 0.2,
    'small.en': 0.7, # TODO: Re-Calibrate a good param
    'medium': 0.2,
    'large': 0.4,
}



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
gpu_tracker.track()




'''Load original Whisper model and dataset'''
# Load audio dataset & ground truth
ds_name = "LibriSpeech.test-other/test_other"
# ds_name = "LibriSpeech.test-clean/test-clean"
ds = load_dataset(ds_name)
# Load original Whisper model (base)
if args.baseline or args.smooth or args.calibrate:
    logger.info(f"Original Whisper {args.model} Loaded")
    model_o = i_whisper.load_model(args.model, download_root="CKPT")
    total_size = sum(p.numel() for p in model_o.state_dict().values())
    logger.info(f"Whisper {args.model} Param Num: {total_size}")
# Decoding options
options = i_whisper.DecodingOptions(fp16=False, language="en")
gpu_tracker.track()

'''Calibration settings'''
smooth_calibrate_samples = 100
quant_calibrate_samples = 15
random.seed(0)


if (args.smooth or args.calibrate):
    '''Load original Whisper model and prepare for quantization'''
    logger.info(f"Whisper_quantized {args.model} Loaded")
    model = i_whisper.load_model_quantized(args.model, download_root="CKPT")
    total_size = sum(p.numel() for p in model.state_dict().values())
    logger.info(f"Whisper_quantized {args.model} Param Num: {total_size}")

    if args.smooth:
        logger.info(f'args.smooth: {args.smooth}, Starting Smooth Routine')

        if args.debug:
            model.quantize_model_debug()
        logger.info(f"Smooth Calibrate START")
        model.observe_channel_max()
        smooth_calibre_loop = tqdm( range(smooth_calibrate_samples) )
        for i in smooth_calibre_loop:
            x = random.randint(0, len(ds['train'])-1)
            smooth_calibre_loop.set_description_str(f"Smooth Calibrate, Data Sample index: {x}")
            logger.debug(f"Smooth Calibrate {i+1} / {smooth_calibrate_samples}, Data Sample index: {x}")
            audio = ds['train'][x]["audio"]["array"]
            audio = np.float32(audio)
            audio = i_whisper.pad_or_trim(audio)
            # mel = whisper.log_mel_spectrogram_fft(audio, n_mels=model.dims.n_mels).to(model.device)
            mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            result = i_whisper.decode(model, mel, options)

        logger.info(f"Smooth Calibrate DONE")
        model.stop_observe_channel_max()

        
        logger.info(f"Smooth START")
        for name, module in model.named_modules():
            if isinstance (module, ResidualAttentionBlock_quantized):
                attn_ln = module.attn_ln 
                qkv = [
                    module.attn.query,
                    module.attn.key,
                    module.attn.value,
                ]
                qkv_input_scales = torch.stack([module.attn.query.running_max_c,module.attn.query.running_max_c,module.attn.query.running_max_c,], dim=0).amax(dim=0)
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha=ln_fcs_alpha[args.model]) 

                mlp_ln = module.mlp_ln
                mlp0 = module.mlp[0]
                mlp2 = module.mlp[2]
                gelu = module.mlp[1]

                mlp0_scales = module.mlp[0].running_max_c
                mlp2_scales = module.mlp[2].running_max_c

                smooth_gelu_fcs(mlp0, gelu, mlp2, mlp2_scales, alpha=gelu_fcs_alpha[args.model])
                logger.debug(f"Smooth {name} DOME")
        logger.info(f"Smooth DONE")


    if args.calibrate:
        if args.debug:
            model.quantize_model_debug()
        logger.info(f"args.calibrate: {args.calibrate}, Starting Quantizaion Routine")
        model.model_open_calibrate()
        quant_calibre_loop = tqdm( range(quant_calibrate_samples) )
        for i in quant_calibre_loop:
            x = random.randint(0, len(ds['train'])-1)
            
            audio = ds['train'][x]["audio"]["array"]
            logger.debug(f"Quant  Calibrate {i+1} / {quant_calibrate_samples}, Data Sample index: {x}")
            quant_calibre_loop.set_description_str(f"Quant  Calibrate, Data Sample index: {x}")
            
            audio = np.float32(audio)
            audio = i_whisper.pad_or_trim(audio)
            mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            # mel = whisper.log_mel_spectrogram_fft(audio, n_mels=model_o.dims.n_mels).to(model_o.device)
            result = i_whisper.decode(model, mel, options)

        logger.info(f"Quant Calibrate DONE")
        model.model_close_calibrate()
        logger.info(f"Quantize Model START")
        model.quantize_model(logger)
        logger.info(f"Quantize Model DONE")
        # if args.save_model:
        #     CKPT_PATH = os.path.join('CKPT', 'consolidated.quant.'+args.model+'.pth')
        #     logger.info(f"args.save_model: {args.save_model}, Saving Quantized Model to {CKPT_PATH}")
        #     if not os.path.exists('CKPT'):
        #         os.makedirs('CKPT')
        #     torch.save(model.state_dict(), CKPT_PATH)

    if model.check_quantized(logger):
        logger.info("MODEL FULL QUANT CHECK: PASS")
    else:
        logger.warning("MODEL FULL QUANT CHECK: FAIL")
else:
    '''Load quantized Whisper model'''
    model = i_whisper.load_model_quantized("base", device=args.device, quantize=True)
    if model.check_quantized(logger):
        logger.info("MODEL FULL QUANT CHECK: PASS")
    else:
        logger.warning("MODEL FULL QUANT CHECK: FAIL")
    CKPT_PATH = os.path.join('CKPT', 'consolidated.quant.'+args.model+'.pth')
    logger.info(f"args.save_model: {args.save_model}, loading Quantized Model from model_quantized.pth")
    if not os.path.exists('CKPT'):
        print("CKPT folder not found!")
    if not os.path.exists(os.path.join('CKPT', 'consolidated.quant.'+args.model+'.pth')):
        print(f"{CKPT_PATH} not found in CKPT folder!")
    state_dict = torch.load(CKPT_PATH, map_location='cpu')
    for key, value in list(state_dict.items()):
        if value.shape == torch.Size([]) and model.state_dict()[key].shape == torch.Size([1]):
            logger.debug(f"mismatched shape for {key}, unsqueezing loaded value")
            state_dict[key] = value.unsqueeze(0)
    log = model.load_state_dict(state_dict=state_dict, strict=False)

if torch.cuda.is_available():
    gpu_tracker.clear_cache()
    gpu_tracker.track()

if (not args.baseline) and (args.smooth or args.calibrate):
    logger.info(f"args.baseline: {args.baseline}, Releasing Original Whisper Model from GPU Memory")
    model_o.cpu()
    model_o = None

if torch.cuda.is_available():
    gpu_tracker.clear_cache()
    gpu_tracker.track()

PreProcessor = i_whisper.audio.PreProcessor(device="cpu", n_mels=model.dims.n_mels)
try:
    PreProcessor.load_param()
except:
    PreProcessor = i_whisper.audio.PreProcessor(device="cpu", n_mels=model.dims.n_mels, param_init=True) 
    PreProcessor.save_param()

if args.test:

    TEXT_GT = []
    TEXT = []
    if args.baseline:
        TEXT_O = []
    if args.fakequant:
        
        '''Test results collection directory'''
        RESULT_HOME = os.path.join('MODEL_TEST_REPOERT')
        if not os.path.exists(RESULT_HOME):
            os.makedirs(RESULT_HOME)

        '''quantized model test log file'''
        RESULT_QUANT = os.path.join(RESULT_HOME, 'quant_'+args.model+'.report')
        f = open(RESULT_QUANT, 'w') # clear previous contents
        f.close()
        f = open(RESULT_QUANT, 'a')
        print("="*50+" R E P O R T "+"="*50, file=f)
        print('Setups:', file=f)
        print('> Online-Quant Model (FP Scale)', file=f)
        print('> INT-FFT', file=f)
        print('> INT-MEL', file=f)
        print('> INT LOG', file=f)

        if args.baseline:
            RESULT_ORIGIN = os.path.join(RESULT_HOME, 'origin_'+args.model+'.report')
            f_o = open(RESULT_ORIGIN, 'w') # clear previous contents
            f_o.close()
            f_o = open(RESULT_ORIGIN, 'a')
            print("="*50+" R E P O R T "+"="*50, file=f_o)
            print('> Baseline Model (FP)', file=f_o)
            print('Setups:', file=f_o)
            print('> INT-FFT', file=f_o)
            print('> INT-MEL', file=f_o)
            print('> INT LOG', file=f_o)

        audio_lengths = []
        test_loop = tqdm( range(len(ds['train']) if args.sample is None else args.sample) )
        for i in test_loop:
            audio = ds['train'][i]["audio"]["array"]

            audio_lengths.append(audio.shape[0]/16000)
            audio = torch.from_numpy(i_whisper.pad_or_trim(audio))  # INT16 range
            # mel = PreProcessor.log_mel_spectrum(audio).to(model.device)
            # mel = i_whisper.log_mel_spectrogram_fft(audio, n_mels=model.dims.n_mels).to(model.device)
            mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

            # Reading Ground Truth
            AUDIO_PATH = ds['train'][i]["audio"]["path"]
            GT_PATH = '-'.join(AUDIO_PATH.split('-')[:-1]) + '.trans.txt'
            GT_FILE = open(GT_PATH, 'r')
            GT_READ = GT_FILE.readlines()[int( AUDIO_PATH.split('-')[-1].split('.')[0] )].split(' ', 1)[1].strip('\n')
            # add GT to TEXT_GT list
            TEXT_GT.append(GT_READ)
            GT_FILE.close()
            test_loop.set_description_str(f"Test Sample {i}:{GT_PATH}")

            # quantized model transcription
            result = i_whisper.decode(model, mel, options)
            # add result to TEXT list
            TEXT.append(result.text)

            if args.baseline:
                # original model transcription
                result_o = i_whisper.decode(model_o, mel, options)
                # add result to TEXT_O list
                TEXT_O.append(result_o.text)


        Align_and_Report(TEXT_GT, TEXT, audio_lengths, f)
        f.close()
    
    if args.baseline:
        # String Processing of fp_model output
        Align_and_Report(TEXT_GT, TEXT_O, audio_lengths, f_o)
        f_o.close()
    
    """TEST INTEGER ONLY INFERENCE VERSION"""
    model.convert_to_int_mode()
    model_new = Whisper_int(model.dims).to(model.device)
    result = model_new.load_state_dict(model.state_dict(), strict=False)
    logger.info("Integer Only Model Loaded from Quantized Model")
    logger.info(f"Missing keys not in Quantized Model: {len(result.missing_keys)}, showing up to 10: {result.missing_keys[:10]}")
    logger.info(f"Unexpected keys in Quantized Model but not in Model: {len(result.unexpected_keys)}, showing up to 10: {result.unexpected_keys[:10]}")
    model = model_new
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        
    '''Integer Only Inference Checker'''
    audit = IntegerOnlyAudit()
    audio = ds['train'][0]["audio"]["array"]
    audio = torch.from_numpy(i_whisper.pad_or_trim(audio))  # INT16 range
    mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    mel_test = mel.mul(model.encoder.input_alpha).round().clamp(-127, 127).to(torch.int8).unsqueeze(0)
    tokens: Tensor = torch.tensor([50258, 50259, 50359]).repeat(1, 1)
    with audit:
        feature = model.encoder(mel_test)
        # set decoder to no dequantize mode
        model.decoder.dequant_need = False
        logits = model.decoder(tokens.to(feature.device), feature)
        model.decoder.dequant_need = True
        logger.info(f"MODEL INT INFERENCE CHECK: PASS")
    
    

    TEXT_GT = []
    TEXT = []
    audio_lengths = []
    test_loop = tqdm( range(len(ds['train']) if args.sample is None else args.sample) )
    for i in test_loop:
        audio = ds['train'][i]["audio"]["array"]

        audio_lengths.append(audio.shape[0]/16000)
        audio = torch.from_numpy(i_whisper.pad_or_trim(audio))  # INT16 range
        # mel = PreProcessor.log_mel_spectrum(audio).to(model.device)
        # mel = i_whisper.log_mel_spectrogram_fft(audio, n_mels=model.dims.n_mels).to(model.device)
        mel = i_whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # Reading Ground Truth
        AUDIO_PATH = ds['train'][i]["audio"]["path"]
        GT_PATH = '-'.join(AUDIO_PATH.split('-')[:-1]) + '.trans.txt'
        GT_FILE = open(GT_PATH, 'r')
        GT_READ = GT_FILE.readlines()[int( AUDIO_PATH.split('-')[-1].split('.')[0] )].split(' ', 1)[1].strip('\n')
        # add GT to TEXT_GT list
        TEXT_GT.append(GT_READ)
        GT_FILE.close()
        test_loop.set_description_str(f"Test Sample {i}:{GT_PATH}")

        # quantized model transcription
        result = i_whisper.decode(model, mel, options)
        # add result to TEXT list
        TEXT.append(result.text)
        
        
    
    '''Test results collection directory'''
    RESULT_HOME = os.path.join('MODEL_TEST_REPOERT')
    if not os.path.exists(RESULT_HOME):
        os.makedirs(RESULT_HOME)

    '''quantized model test log file'''
    RESULT_INT = os.path.join(RESULT_HOME, 'int_'+args.model+'.report')
    f = open(RESULT_INT, 'w') # clear previous contents
    f.close()
    f = open(RESULT_INT, 'a')
    print("="*50+" R E P O R T "+"="*50, file=f)
    print('Setups:', file=f)
    print('> Static Integer Only Model', file=f)
    print('> INT-FFT', file=f)
    print('> INT-MEL', file=f)
    print('> INT LOG', file=f)
    Align_and_Report(TEXT_GT, TEXT, audio_lengths, f)
    f.close()
    logger.info("Integer Only Model Inference Test Done, please check the report in MODEL_TEST_REPOERT folder")
    
    if args.save_model:
        logger.info("Saving Integer Only Model to CKPT folder")

        i_whisper.save_int_model(
            model,
            prefix="CKPT",
            model_name=args.model,
        )

        loaded_model = i_whisper.load_int_model(
            args.model,
            device="cpu",
        )

        # 1. Check serialized persistent state
        i_whisper.verify_int_model_roundtrip(
            model,
            loaded_model,
            logger.info
        )

        logger.info(
            "Integer Only Model save/load round-trip verification passed"
        )
    
    
else:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of int_model parameters: {total_params}")

    '''Load Audio & PreProcessing'''
    audio = i_whisper.load_audio(os.path.join('vcsdl_dataset', args.filename))
    print(f"Original audio length: {audio.shape[0]/16000:.2f} seconds")
    audio_list = []
    while audio.shape[0] > 30*16000:
        # cut the audio to many 30-second clips if too long
        audio_list.append( audio[:30*16000] )
        audio = audio[30*16000:]
    # the last clip (less than 30 seconds)
    audio = i_whisper.pad_or_trim(audio)
    audio_list.append(audio)
    del audio
    print(f"Original Audio is cut into {len(audio_list)} 30-second clips: ")
    for idx, audio in enumerate(audio_list):
        mel = PreProcessor.log_mel_spectrum(audio).to(model.device)
        result = i_whisper.decode(model, mel, options)
        print(f"clip{idx}: {result.text}")
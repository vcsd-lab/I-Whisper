#!/bin/bash
# python3 quantize.py --calibrate --smooth --test --model tiny --fakequant
# python3 quantize.py --calibrate --smooth --test --model base --fakequant
# python3 quantize.py --calibrate --smooth --test --model small --fakequant
# python3 quantize.py --calibrate --smooth --test --model medium --fakequant
# python3 quantize.py --calibrate --smooth --test --model large --fakequant

python3 int_transcribe.py --test --device cpu --model tiny
python3 int_transcribe.py --test --device cpu --model base
python3 int_transcribe.py --test --device cpu --model small
python3 int_transcribe.py --test --device cpu --model medium
python3 int_transcribe.py --test --device cpu --model large

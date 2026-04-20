#!/bin/bash
python3 quantize.py --calibrate --smooth --test --model tiny --fakequant
python3 quantize.py --calibrate --smooth --test --model base --fakequant
python3 quantize.py --calibrate --smooth --test --model small --fakequant
python3 quantize.py --calibrate --smooth --test --model medium --fakequant
python3 quantize.py --calibrate --smooth --test --model large --fakequant

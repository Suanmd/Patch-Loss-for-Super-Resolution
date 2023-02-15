#!/bin/bash

cuda=1
model_name='0'
iter=500000
scale=4

num=$[scale/2]
CUDA_VISIBLE_DEVICES="$cuda" python inference/inference_swinir.py --input datasets/Set5/LRbicx"$scale" --scale "$scale" --patch_size 48 --model_path experiments/00"$num"_"$model_name"_SwinIR_SRx"$scale"_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_"$iter".pth --output results/SwinIRx"$scale"_"$model_name"/Set5
CUDA_VISIBLE_DEVICES="$cuda" python inference/inference_swinir.py --input datasets/Set14/LRbicx"$scale" --scale "$scale" --patch_size 48 --model_path experiments/00"$num"_"$model_name"_SwinIR_SRx"$scale"_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_"$iter".pth --output results/SwinIRx"$scale"_"$model_name"/Set14
CUDA_VISIBLE_DEVICES="$cuda" python inference/inference_swinir.py --input datasets/BSDS100/LRbicx"$scale" --scale "$scale" --patch_size 48 --model_path experiments/00"$num"_"$model_name"_SwinIR_SRx"$scale"_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_"$iter".pth --output results/SwinIRx"$scale"_"$model_name"/BSDS100
CUDA_VISIBLE_DEVICES="$cuda" python inference/inference_swinir.py --input datasets/urban100/LRbicx"$scale" --scale "$scale" --patch_size 48 --model_path experiments/00"$num"_"$model_name"_SwinIR_SRx"$scale"_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_"$iter".pth --output results/SwinIRx"$scale"_"$model_name"/urban100
CUDA_VISIBLE_DEVICES="$cuda" python inference/inference_swinir.py --input datasets/manga109/LRbicx"$scale" --scale "$scale" --patch_size 48 --model_path experiments/00"$num"_"$model_name"_SwinIR_SRx"$scale"_scratch_P48W8_DIV2K_500k_B4G8/models/net_g_"$iter".pth --output results/SwinIRx"$scale"_"$model_name"/manga109

CUDA_VISIBLE_DEVICES="$cuda" python scripts/metrics/calculate_metrics.py --gt datasets/Set5/GTmod12/ --restored results/SwinIRx"$scale"_"$model_name"/Set5 --crop_border "$scale"
CUDA_VISIBLE_DEVICES="$cuda" python scripts/metrics/calculate_metrics.py --gt datasets/Set14/GTmod12/ --restored results/SwinIRx"$scale"_"$model_name"/Set14 --crop_border "$scale"
CUDA_VISIBLE_DEVICES="$cuda" python scripts/metrics/calculate_metrics.py --gt datasets/BSDS100/GTmod12/ --restored results/SwinIRx"$scale"_"$model_name"/BSDS100 --crop_border "$scale"
CUDA_VISIBLE_DEVICES="$cuda" python scripts/metrics/calculate_metrics.py --gt datasets/urban100/GTmod12/ --restored results/SwinIRx"$scale"_"$model_name"/urban100 --crop_border "$scale"
CUDA_VISIBLE_DEVICES="$cuda" python scripts/metrics/calculate_metrics.py --gt datasets/manga109/GTmod12/ --restored results/SwinIRx"$scale"_"$model_name"/manga109 --crop_border "$scale"
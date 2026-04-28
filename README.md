# Sonar_KD_Ultra_Net

This repository provides supplementary reproducibility materials for the manuscript:

"Lightweight Forward-Looking Sonar Sensing Framework for Embedded Target Detection in Resource-Constrained Underwater Systems"

## Contents

- `ph_model.onnx`: exported ONNX model of the proposed lightweight student detector.
- `onnx_inference.py`: Python script for running ONNX inference.
- `Experimental verification file/`: demo images from UATD-Test-1 for quick testing.
- `Comparative experiments with SOTA/`: comparison-model weights or result files used for experimental verification.
- `ONNX_VIDEO/`: supplementary materials related to ONNX inference and video demonstration.
- `Proof of experiments requested by reviewers to supplement/`: additional materials prepared in response to reviewer comments.

## Notes

The ONNX model is provided for inference verification and quick reproduction of demo results on UATD-Test-1 images. Minor numerical differences may occur between the original PyTorch model and the exported ONNX model due to differences in inference backends and floating-point computation.

The public UATD dataset can be downloaded from its original source. The Zhanjiang Bay No.1 field data are subject to institutional data-management restrictions. Selected watermarked demonstration materials can be provided for review purposes upon reasonable request.

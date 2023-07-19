# IoT-V2E

PyTorch implementation on: IoT-V2E: an uncertainty-aware cross-modal hashing retrieval between infrared-videos and EEGs for automated sleep state analysi

## Introduction
we propose a novel IoT system (IoT-V2E) for monitoring and analysing sleep state at home using ubiquitous IR visual camera sensors. 

Specifically, IoT-V2E is a uncertainty-aware cross-modal hashing retrieval system that finds the most similar EEG signal representation in a database, rather than using a sleep stage classification paradigm, for sleep stage inference and analysis.

<p align="center">
<img src="https://github.com/SPIresearch/IoT-V2E/blob/main/NEWSYS.png" width="70%">
</p>

## Getting Started
### Requirmenets:
- python >= 3.6.10 
- pytorch >= 1.1.0
- FFmpeg, FFprobe
- Numpy
- Sklearn
- Pandas
- openpyxl
- mne=='0.20.7'

<p align="center">
<img src="https://github.com/SPIresearch/IoT-V2E/blob/main/Methods.png" width="70%">
</p>



###Train：
Run trainso.py  (single-GPU training)

You need to input some parameters based on your own settings, including hash code length, the location of the cross-modal dataset, and the location of generated hash codes, uncertainties, and checkpoints."

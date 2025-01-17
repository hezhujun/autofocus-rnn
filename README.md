# Autofocus-RNN
[中文](README-CN.md)

## Introduction
Implementation of 《Autofocus of whole slide imaging based on convolution and recurrent neural
networks》.

## Requirement
- PyTorch 1.1.0
- PIL
- OpenCV2

## Dataset
Baida Pan link: https://pan.baidu.com/s/1w8P_1iloZrqw-XeeuTUooQ Extraction code: nn2u

## Model Parameters
Baida Pan link: https://pan.baidu.com/s/1bZfugCtaq83EkUlpwp1QEA Extraction code: bqf8

## Usage Guide
### Dataset Processing
After downloading and extracting the dataset, utilize the tools in the `dataset/tools` directory to convert images into the structure required for training.

1. Construct the `focus_measures` tool in the `dataset/tools/focus_measures` directory, which relies on OpenCV2 and CMake as the build tool.

2. Use the Python scripts located in the `dataset/tools` directory to generate JSON files recording dataset information. The `calc_focus_measures.py` script employs the tool created in Step 1 to compute focus measures and saves the data in JSON files for ease of use during model training.

### Training/Evaluating the Model
Configure `config.py`, primarily setting the dataset path and specifying the training, validation, and testing datasets.

Execute `train.py` or `evaluate.py` to train or test model.

## Note
This project is no longer maintained. After several years, the author finds it challenging to recall the implementation details of the code. 🐶

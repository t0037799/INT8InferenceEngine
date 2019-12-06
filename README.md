# INT8InfernceEngine
An inference engine for 8-bit quantized neural network

## requirement
python
* torch
* torchvision
* numpy

lib
* mkl

## Benchmark

|Batch size|Pytorch FP32|Pytorch INT8|Gains|My Engine INT8|Gains|
|---|---|---|---|---|---|
|1|
|100|
|10000|


## Current Progress

* Hard coded fully-connected layer function
* Import pytorch model's parameters through numpy
* Hard coded quantization parameters
* Simple benchmark

## TODO

* oop design for entire neural network
* convolution layer, relu, maxpool layers
* unit test on each kernel and layer
* optimization of quantized kernel
* auto calibration for quantization
* quantized model save/load

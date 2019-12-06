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
*Testing on mnist with only one fully connected layer*

original accuracy: 92%

pytorch int8 accuracy: 92.6%

my engine accuracy: 91.3%

|Batch size|Pytorch FP32|Pytorch INT8|Gains|My Engine INT8|Gains|
|---|---|---|---|---|---|
|1|561ms|576ms|0.97|617ms|0.91|
|100|9.3ms|12.6ms|0.74|16.1ms|0.58|
|10000|2.3ms|3.8ms|0.71|5.7ms|0.47|


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

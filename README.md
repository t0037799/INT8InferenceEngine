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
*Testing on cifar10 with alexnet*

*cpu: i9-9900k*

original accuracy: 77.8%

pytorch int8 accuracy: 77.4%

my engine int8 accuracy: 76.1%

|Batch size|Pytorch FP32|Pytorch INT8|My Engine FP32|My Engine INT8|
|---|---|---|---|---|
|10|50.4s|28.6s|1m22s|2m9s|
|100|37s|23.9s|57.8s|1m5s|
|1000|37.9ms|27.4ms|55.3s|56s|


## Current Progress

* Support all operators for AlexNet
* Import pytorch model's parameters through numpy
* Automatic quantization calibrator
* Simple benchmark
* Basic OOP design

## TODO

* Document
* unit test on each kernel and layer

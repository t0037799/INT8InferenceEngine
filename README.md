# INT8InfernceEngine
An inference engine for 8-bit quantized neural network

The computing kernel design leverage intel mkl gemm and openmp to parallize the procedure.

The design of interface mimic pytorch's in order to make pytorch user get started quickly.

Other miscs, thanks to numpy array interface, you can transfer data between pytorch, numpy and this inference engine with little overhead.
Also, something like data loading/preprocessing could be done in pytorch or numpy to reduce repeated work.

## requirement
python
* torch
* torchvision
* numpy
* pybind11

lib
* mkl
* openmp

## Installation troubleshooting
* **missing pybind11:** git clone https://github.com/pybind/pybind11.git to your repo or change add_subdirectory in CMakeLists to your pybind11 root directory
* **missing mkl:** try to get library from intel official webcite https://software.intel.com/en-us/mkl/choose-download or using conda to install. After that, remember to set ${MKLROOT} environment variable for cmake to get path of mkl.

## Benchmark
*Testing on resized 224x224 cifar10 with alexnet*

*cpu: i9-9900k*

original accuracy: 77.8%

pytorch int8 accuracy: 77.4%

my engine int8 accuracy: 76.1%

|Batch size|Pytorch FP32|Pytorch INT8|My Engine FP32|My Engine INT8|
|---|---|---|---|---|
|10|50.4s|28.6s|1m16s|1m2s|
|100|37s|23.9s|48.3s|36.6s|
|1000|37.9s|27.4s|45.9s|34.2s|

*Testing with different size of network and input data(100 minibatch)*

|Model/Data| Pytorch FP32|My Engine FP32|My Engine INT8|
|---|---|---|---|
|one fully connected/mnist|9ms|3.76ms|19.6ms|
|simple convolution/cifar10|1.29s|1.43s|1.39s|

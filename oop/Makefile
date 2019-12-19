CXXFLAGS = -march=native -O3 -DMKL_ILP64 -m64
CXXFLAGS += -fvisibility=hidden -fopenmp -std=c++17
ifeq ($(shell uname -r), 5.3.6-arch1-1-ARCH)
MKLROOT = /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl
MKLINC = ${MKLROOT}/include
MKLLIB = ${MKLROOT}/lib/intel64
else
MKLROOT = ${HOME}/opt/conda
MKLINC = ${MKLROOT}/include
MKLLIB= ${MKLROOT}/lib
endif

tensor_core.so: pybind11.cc layer.h tensor.h conv2d.h calibrator.h quant_utils.h
	g++ -o $@ $< -fPIC -shared \
		$(CXXFLAGS)  \
		-I$(MKLINC) `python3 -m pybind11 --includes`\
		-L$(MKLLIB) \
		-Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl


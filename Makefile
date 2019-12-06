# modifiy MKL related params if link error

CXXFLAGS = -march=native -O3 -DMKL_ILP64 -m64
ifeq ($(shell uname -r), 5.3.6-arch1-1-ARCH)
MKLROOT = /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl
MKLINC = ${MKLROOT}/include
MKLLIB = ${MKLROOT}/lib/intel64
else
MKLROOT = ${HOME}/opt/conda
MKLINC = ${MKLROOT}/include
MKLLIB= ${MKLROOT}/lib
endif

all: i8ie.so 

i8ie.so : i8ie.cc 
	$(CXX) -o $@ -fPIC -shared \
		-I$(MKLINC) `python3 -m pybind11 --includes`\
		$(CXXFLAGS) $^ \
		-L$(MKLLIB) \
		-Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl
clean:
	rm i8ie.so

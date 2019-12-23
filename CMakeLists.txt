cmake_minimum_required(VERSION 2.8.12)
project(i8ie)

include_directories(include)
set(SRC_FILES
    src/pybind11.cc
    src/functional.cc
    src/quantize_utils.cc
    src/calibrator.cc
    src/layer.cc
    src/fully_connected.cc
    src/conv2d.cc)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(RT_LIB mkl_rt)
set(MKLROOT /opt/intel/compilers_and_libraries_2019.5.281/linux/mkl)
find_library(
  MKL_RT_LIBRARY
  NAMES ${RT_LIB}
  PATHS $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64 $ENV{INTEL}/mkl/lib/intel64
  NO_DEFAULT_PATH)

find_path(
  MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS $ENV{MKLROOT}/include)

add_subdirectory(pybind11)

pybind11_add_module(_CXX_i8ie ${SRC_FILES})
if(MKL_RT_LIBRARY AND MKL_INCLUDE_DIR)
  target_link_libraries(i8ie INTERFACE mkl_rt)
  include_directories(${MKL_INCLUDE_DIR})
else()
  message(
    SEND_ERROR
      "MKL library not found. Please set \${MKLROOT} env and check if MKL library installed"
  )
endif()

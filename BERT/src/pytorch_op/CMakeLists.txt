# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET BertOpPT)

file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH)
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})

find_package(Torch)
find_package(dnnl REQUIRED)

# NOTE: (krzychut)
# Not sure why, but libBertOpPT.so wants to link to libmkl_intel_thread.so.2, even if I remove DNNL from this target.
# Without linking to MKL libs, I get an undefined symbol error from the BertOp.
# None of the pytorch libs (libtoch.so, libtoch_cpu.so etc.) or other libs reported by `ldd` seem to link to MKL. 
find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    ${TORCH_INCLUDE_DIR}
)

add_library(${TARGET} SHARED ${SOURCES})

set_target_properties(${TARGET} PROPERTIES
    CXX_STANDARD_REQUIRED TRUE
    CXX_STANDARD 14)

target_link_libraries(${TARGET} PUBLIC BertLayer)

target_link_libraries(${TARGET} PUBLIC DNNL::dnnl)

target_link_libraries(${TARGET} PUBLIC ${TORCH_LIBRARIES})

target_link_libraries(${TARGET} PUBLIC MKL::MKL)

target_link_libraries(${TARGET} PUBLIC ${OpenMP_gomp_LIBRARY} PUBLIC ${OpenMP_pthread_LIBRARY})
target_compile_options(${TARGET} PUBLIC ${OpenMP_CXX_FLAGS})

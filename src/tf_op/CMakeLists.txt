# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET BertOp)

file(GLOB HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(dnnl)

find_package( OpenMP REQUIRED )
add_compile_options("${OpenMP_CXX_FLAGS}")

find_package (Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow; print(tensorflow.sysconfig.get_include())"                       OUTPUT_VARIABLE TF_INCLUDE)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')"    OUTPUT_VARIABLE TF_LFLAGS)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')" OUTPUT_VARIABLE TF_CFLAGS)

set(CMAKE_CXX_FLAGS "-std=c++14 -liomp5 -Wall -Wextra -Wno-comment ${TF_CFLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed ${TF_LFLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
include_directories(SYSTEM ${TF_INCLUDE})

add_library(${TARGET} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC ${OpenMP_gomp_LIBRARY} PUBLIC ${OpenMP_pthread_LIBRARY})
target_link_libraries(${TARGET} PUBLIC BertLayer)
target_compile_options(${TARGET} PUBLIC ${OpenMP_CXX_FLAGS})

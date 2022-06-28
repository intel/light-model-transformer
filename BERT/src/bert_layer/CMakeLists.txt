# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET BertLayer)

file(GLOB HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(dnnl)

find_package( OpenMP REQUIRED )
add_compile_options("${OpenMP_CXX_FLAGS}")

add_library(${TARGET} ${SOURCES} ${HEADERS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC DNNL::dnnl)
target_link_libraries(${TARGET} PUBLIC ${OpenMP_gomp_LIBRARY} PUBLIC ${OpenMP_pthread_LIBRARY})
target_compile_options(${TARGET} PUBLIC ${OpenMP_CXX_FLAGS})

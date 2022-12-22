# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET BertLayer)

file(GLOB HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(dnnl)

find_package( OpenMP REQUIRED )
add_compile_options("${OpenMP_CXX_FLAGS}")

add_library(${TARGET} INTERFACE)
target_include_directories(${TARGET} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} INTERFACE DNNL::dnnl)

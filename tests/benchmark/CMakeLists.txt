# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET benchmark)

file(GLOB HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp)
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    message("using Intel C++")
    set(CMAKE_CXX_FLAGS "-std=c++14 -parallel ${CMAKE_CXX_FLAGS}")
endif()

add_executable(${TARGET} ${SOURCES} ${HEADERS})

target_link_libraries(${TARGET} PRIVATE BertLayer)

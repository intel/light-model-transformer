# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(model_modifier VERSION ${CMAKE_PROJECT_VERSION})

find_package (Python3 COMPONENTS Interpreter REQUIRED)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/model_modifier/__init__.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
                   COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} build
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                   DEPENDS ${DEPS}
)

add_custom_target(PythonPackageSetup ALL DEPENDS ${OUTPUT})

# We want protoc to output the generated files to the same directory as
# the python sources copied by setup.py
add_subdirectory(proto ${CMAKE_CURRENT_BINARY_DIR}/build/lib)

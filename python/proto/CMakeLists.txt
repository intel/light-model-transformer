# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED Tensorflow_PROTO)
    message(FATAL_ERROR "Tensorflow proto path is required to build this project's protocol buffers. "
        "Typically, this will be the include directory of the tensorflow python package.")
endif()

find_package(Protobuf REQUIRED)

set(Protobuf_IMPORT_DIRS
    ${Tensorflow_PROTO}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

file(GLOB_RECURSE Protobuf_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.proto)
message("Proto files: ${Protobuf_FILES}")

# This function internally converts the files to absolute paths.
# Protobuf compiler is then called from the build space, with the above imports
# and absolute paths to files. In consequence, the python descriptors are
# incorrectly generated and cannot import one another.
protobuf_generate_python(Protobuf_OUTPUTS ${Protobuf_FILES})


# Attempt at a custom command to run protoc correctly.

# Generate protoc flags from import dirs
# foreach(_dir ${Protobuf_IMPORT_DIRS})
#     list(APPEND Protobuf_INCLUDE_PATHS "-I${_dir}")
# endforeach()

# Generate output names to provide as custom command output.
# foreach(_file ${Protobuf_FILES})
#     get_filename_component(_filename ${_file} NAME_WLE)
#     get_filename_component(_dir ${_file} DIRECTORY)
#     list(APPEND Protobuf_OUTPUTS "${_dir}/${_filename}_pb2.py")
# endforeach()

# set(DEPS "${Protobuf_FILES}" "protobuf::protoc")
# add_custom_command(
#     OUTPUT ${Protobuf_OUTPUTS}
#     COMMAND # TODO: Recreate the .proto files and directory tree in the build space
#     COMMAND protobuf::protoc --python_out=${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_PATHS} ${Protobuf_FILES}
#     DEPENDS ${DEPS}
#     COMMENT "Running protocol buffer compiler on ${Protobuf_FILES}"
#     VERBATIM )


add_custom_target(PythonProtos ALL DEPENDS ${Protobuf_OUTPUTS})

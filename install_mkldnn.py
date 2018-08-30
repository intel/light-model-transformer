#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================
import os
import subprocess
import argparse

from string import Template

MKLDNN_VERSION = 'v0.16'
MKLDNN_PATH = 'third_party/mkl-dnn'

def exec_cmd(cmd, title, check_output=True):
    print(title)
    print(cmd)
    if check_output:
        try:
            cmd = cmd.split(' ')
            output = subprocess.check_output(cmd)
            return output
        except subprocess.CalledProcessError as e:
            print('\n    ERROR: ', e.output)
            exit()
    else:
        os.system(cmd)


def cpu_flag_contains_avx512():
    r = os.system("cat /proc/cpuinfo | grep flags | grep avx512 &> /dev/null")

    if r == 0: return True
    return False


def is_gcc_support_avx512():
    str_version = exec_cmd("gcc -dumpversion", "Check gcc version")
    versions = str_version.decode().split('.')
    if int(versions[0]) > 4:
        return True
    if int(versions[0]) == 4 and int(versions[1]) >= 9:
        return True
    return False


def make_patch():
    cpu_type_str, cmake_patch_str = "not_avx512", ""

    if is_gcc_support_avx512() and cpu_flag_contains_avx512():
        cpu_type_str = "avx512"
        cmake_patch_str += "[cmake/platform.cmake]\n"
        cmake_patch_str += "+ among(\"elseif(UNIX OR APPLE OR MINGW)\", \"if(CMAKE_CXX_COMPILER_ID\")\n"
        cmake_patch_str += "    set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -mavx512f\")"

    with open("cfg/PATCH.cfg", "r") as f:
        data = f.read()
    data = Template(data).safe_substitute( \
            cpu_type = cpu_type_str,
            cmake_patch = cmake_patch_str)
    with open("PATCH/mkldnn_patch/PATCH.CFG", "w") as f:
        f.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mkldnn_path", "-p", default=MKLDNN_PATH, type=str, help="Path of mkl-dnn, default:[%s]" % MKLDNN_PATH)
    args = parser.parse_args()

    download_mkldnn = 'git clone https://github.com/intel/mkl-dnn -b %s %s' % (MKLDNN_VERSION, args.mkldnn_path)
    apply_patch = 'python PATCH/mkldnn_patch/apply_patch.py --path=%s' % args.mkldnn_path
    install_mkldnn = 'mkdir -p %s/build && cd %s/build && cmake .. && make -j && make install' % (args.mkldnn_path, args.mkldnn_path)

    if os.path.exists(args.mkldnn_path):
        print("  Exist path['third_party/mkl-dnn'], jump the step of download mkldnn.")
    else:
        exec_cmd(download_mkldnn, "Cloning 'mkl-dnn' into 'third_party/mkl-dnn'...")

    make_patch()
    exec_cmd(apply_patch, "Patch mkl-dnn ...")
    exec_cmd(install_mkldnn, "Make and Install mkl-dnn ...", False)

    print('\nDone.\n')

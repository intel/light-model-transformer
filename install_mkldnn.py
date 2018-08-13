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

MKLDNN_VERSION = 'v0.15'
MKLDNN_PATH = 'third_party/mkl-dnn'

def exec_cmd(cmd, title, check_output=True):
    print(title)
    print(cmd)
    if check_output:
        try:
            cmd = cmd.split(' ')
            output = subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            print('\n    ERROR: ', e.output)
            exit()
    else:
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mkldnn_path", "-p", default=MKLDNN_PATH, type=str, help="Path of mkl-dnn, default:[%s]" % MKLDNN_PATH)
    args = parser.parse_args()

    download_mkldnn = 'git clone https://github.com/intel/mkl-dnn -b %s %s' % (MKLDNN_VERSION, args.mkldnn_path)
    apply_patch = 'python PATCH/mkldnn_patch/apply_patch.py --path=%s' % args.mkldnn_path
    install_mkldnn = 'mkdir -p %s/build && cd %s/build && cmake .. && make -j && make install' % (args.mkldnn_path, args.mkldnn_path)

    if os.path.exists(args.mkldnn_path):
        print("  Exist path['third_party/mkl-dnn'], jump the step of install mkldnn.")
    else:
        exec_cmd(download_mkldnn, "Cloning 'mkl-dnn' into 'third_party/mkl-dnn'...")

    exec_cmd(apply_patch, "Patch mkl-dnn ...")
    exec_cmd(install_mkldnn, "Make and Install mkl-dnn ...", False)

    print('\nDone.\n')

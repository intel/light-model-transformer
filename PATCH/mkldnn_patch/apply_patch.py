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
import argparse

def check_path(path):
    if not os.path.exists(path):
        print("ERROR: [%s] not exit" % path)
        exit()

def exe_move(mkldnn_path, move_info):
    print("  Moving ...")
    for from_dir in move_info.keys():
        target_dir = '%s/%s' % (mkldnn_path, from_dir)
        print("    %s" % target_dir)

        check_path(target_dir)

        for elem in move_info[from_dir]:
            os.system("cp %s/%s/%s %s" % (os.path.dirname(__file__), from_dir, elem, target_dir))


def exe_modify(mkldnn_path, modify_info):
    print("  Modifing ...")
    for m_file in modify_info.keys():
        target_file = '%s/%s' % (mkldnn_path, m_file)
        origin_file = '%s_origin' % target_file
        print("    %s" % target_file)

        if not os.path.exists(origin_file): os.system("cp %s %s" % (target_file, origin_file))

        m_entrance, m_exit, m_content = [], [], []
        for index in range(len(modify_info[m_file])):
            m_entrance.append(modify_info[m_file][index][0])
            m_exit.append(modify_info[m_file][index][1])
            m_content.append(modify_info[m_file][index][2:])

        content = ""
        with open(origin_file, "r") as f:
            content = f.read()
        content = content.split("\n")
        len_content = len(content) - 1

        ready_to_modify = False
        modify_num, line_index, comment_out_code_index = -1, -1, -1
        while line_index < len_content:
            line_index += 1
            line = content[line_index]
            if len(line) == 0: continue

            # Is ENTRANCE or not
            if modify_num == -1:
                for index in range(len(m_entrance)):
                    if m_entrance[index] in line:
                        modify_num = index
                # continue no matter is ENTRANCE or not
                continue

            # Got entrance, ready to modify
            # Save space num
            space_num = 0
            for index in range(len(line)):
                if line[index] == " ": space_num += 1
                else: break

            # Save the index of first line of the last comment-out code before EXIT.
            if line.strip()[:2] in ["//", "/*", "# "]:
                if comment_out_code_index == -1:
                    comment_out_code_index = line_index
                continue

            # Confront EXIT, need to modify
            if m_exit[modify_num] == "None":
                if m_entrance[modify_num] not in line:
                    ready_to_modify = True
            else:
                for index in range(len(m_exit)):
                    if m_exit[index] in line:
                        ready_to_modify = True

            if ready_to_modify:
                # Get the index of line need to modify
                modify_line = line_index
                if comment_out_code_index != -1: modify_line = comment_out_code_index

                # Make the new line
                tmp_content = ''
                for elem in m_content[modify_num]:
                    if space_num != 0: tmp_content += " "*space_num
                    tmp_content += "{0}\n".format(elem)
                tmp_content += "\n%s" % content[modify_line]

                # Modify
                content[modify_line] = tmp_content

                # Clean
                ready_to_modify = False
                modify_num = -1
            else:
                # Normal line(not comment-out code) between ENTRANCE and EXIT
                comment_out_code_index = -1
        
        final_content = ""
        for line in content:
            final_content += "{0}\n".format(line)

        with open(target_file, "w") as f:
            f.write(final_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", default="PATCH.CFG", type=str, help="patch configure file")
    parser.add_argument("--path", default="/home/xuesong/recent_work/mkl-dnn", type=str, help="mkl-dnn path")
    args = parser.parse_args()

    mkldnn_path = args.path

    check_path(mkldnn_path)

    content = ""
    with open("%s/%s" % (os.path.dirname(__file__), args.patch), "r") as f:
        content = f.read()

    move = False
    move_info = {} # { target_directory: [file1, file2 ...] }

    modify = False
    modify_file = ""
    modify_entrance, modify_exit = "", ""
    modify_num = -1
    modify_info = {} # { modify_file: [entrace, exit, content1, content2, ...] }

    for line in content.split("\n"):
        line = line.strip()

        if len(line) == 0: continue

        if line[0] == '[':
            if "move" in line:
                move = True
                modify = False
            else:
                move = False
                modify = True
                modify_file = line[1:-1]
                modify_info[modify_file] = []
                modify_num = -1
            continue
            
        if move:
            if line[0] == '>': 
                move_dir = line.split(">")[1].split(":")[0].strip()
                move_info[move_dir] = []
                continue

            file_name = line.strip()
            move_info[move_dir].append(file_name)
            
        elif modify:
            if line[0] == '+':
                modify_entrance = line.split('", "')[0].split('("')[1].strip()
                modify_exit = line.split('", "')[1].split('")')[0].strip()
                modify_info[modify_file].append([modify_entrance, modify_exit])
                modify_num += 1
                continue

            modify_line = line.strip()
            modify_info[modify_file][modify_num].append(modify_line)

    print("\nBegin.")
    exe_move(mkldnn_path, move_info) 
    exe_modify(mkldnn_path, modify_info)
    print("Done.\n")
    print("PS: Origin file backuped from ${file_name} to ${file_name}_origin\n")

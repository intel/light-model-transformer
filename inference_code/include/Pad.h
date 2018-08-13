/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

void get_pad(int in_height, int in_width, int ksize_h, int ksize_w, int stride_h, int stride_w, \
            int* pad_left, int* pad_right, int* pad_top, int* pad_bottom) {
    int pad_along_height = 0, pad_along_width = 0;
    if (in_height % stride_h == 0)
        pad_along_height = max(ksize_h - stride_h, 0);
    else
        pad_along_height = max(ksize_h - (in_height % stride_h), 0);

    if (in_width % stride_w == 0)
        pad_along_width = max(ksize_w - stride_w, 0);
    else
        pad_along_width = max(ksize_w - (in_width % stride_w), 0);

    *pad_top = pad_along_height / 2;
    *pad_bottom = pad_along_height - *pad_top;
    *pad_left = pad_along_width / 2;
    *pad_right = pad_along_width - *pad_left;
}

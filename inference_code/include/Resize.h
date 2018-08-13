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
#include <cv.h>
#include <highgui.h> 
#include <sys/time.h>
#include <iostream>

static void resize_pic(std::string input, std::string output, int w, int h) {
    // load image
	cv::Mat SrcImg = cv::imread(input);
	cv::Size ResImgSiz = cv::Size(w, h);
	cv::Mat ResImg = cv::Mat(ResImgSiz, SrcImg.type());
	cv::resize(SrcImg, ResImg, ResImgSiz, CV_INTER_CUBIC);
	cv::imwrite(output, ResImg);
}

/*
int main(int argc, char **argv) {
	std::string i = "input.jpg";
	std::string o = "input.jpg";
    resize_pic(i, o, 324, 324);
    return 0;
}
*/

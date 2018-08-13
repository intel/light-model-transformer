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
#include "include/Sum.h"
#include "include/Concat.h"
#include "include/Convolution.h"
#include "include/Pooling.h"
#include "include/InnerProduct.h"
#include "include/Softmax.h"
#include "include/Resize.h"
#include "include/Reorder.h"
#include "include/Scales.h"
#include "include/Split.h"
#include "include/BatchNorm.h"
#include "include/ExtractImagePatches.h"

using namespace std;

class Model
{
public:
    Model(const char *_weights_path, int _batch_size, int _height=224, int _width=224, int _channel=3) {
        this->weights_path = _weights_path;
        this->batch_size = _batch_size;
        this->input_height = _height;
        this->input_width = _width;
        this->input_channel = _channel;

        create_net();
        if ( !instance_num ) read_weights();
        init_net();

        instance_num ++;
    }
    ~Model() {
        release_io();
        if ( instance_num == 1 ) release_weights();
        instance_num --;
    }


    float* inference(float *input) {
        src_memory->set_data_handle(input);
        stream(stream::kind::eager).submit(net).wait();
        float* last_output = (float*)Conv2D_135_out->get_data_handle();

        return last_output;
    }


    void create_net() {
        memory::dims src_tz = {batch_size, input_channel, input_height, input_width};
        vector<float> placeholder(batch_size * input_height * input_width * input_channel);
        src_memory = new memory({{{src_tz}, memory::data_type::f32,
                          memory::format::nchw}, cpu_engine}, placeholder.data());

        // origin_layer: layer1/layer1_conv/Conv2D
        Conv2D_1 = new Convolution("Conv2D_1", 64,
                        input_channel, input_height, input_width,
                        7, 7, 2, 3, 2, 3, 2, 2,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/MaxPool2D/MaxPool
        MaxPool_1 = new Pooling("MaxPool_1",
                        Conv2D_1->getOutputChannels(), Conv2D_1->getOutputHeight(), Conv2D_1->getOutputWidth(),
                        3, 3, 0, 1, 0, 1, 2, 2,
                        "Max",
                        Conv2D_1->getQuantizeType());

        // origin_layer: layer2/block0/common_bn_relu/FusedBatchNorm
        BatchNorm_1 = new BatchNorm("BatchNorm_1", 
                        MaxPool_1->getOutputChannels(), MaxPool_1->getOutputHeight(), MaxPool_1->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer2/block0/sub1/sub1_conv/Conv2D
        Conv2D_2 = new Convolution("Conv2D_2", 64,
                        BatchNorm_1->getOutputChannels(), BatchNorm_1->getOutputHeight(), BatchNorm_1->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block0/sub2/sub2_conv/Conv2D
        Conv2D_3 = new Convolution("Conv2D_3", 64,
                        Conv2D_2->getOutputChannels(), Conv2D_2->getOutputHeight(), Conv2D_2->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block0/sub3/sub3_conv/Conv2D
        Conv2D_4 = new Convolution("Conv2D_4", 256,
                        Conv2D_3->getOutputChannels(), Conv2D_3->getOutputHeight(), Conv2D_3->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block0/shortcut/sub_sc/Conv2D
        Conv2D_5 = new Convolution("Conv2D_5", 256,
                        BatchNorm_1->getOutputChannels(), BatchNorm_1->getOutputHeight(), BatchNorm_1->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block0/shortcut/add
        Add_1 = new Sum("Add_1", 
                        Conv2D_4->getOutputChannels(), Conv2D_4->getOutputHeight(), Conv2D_4->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer2/block1/residual_bn_relu/FusedBatchNorm
        BatchNorm_2 = new BatchNorm("BatchNorm_2", 
                        Add_1->getOutputChannels(), Add_1->getOutputHeight(), Add_1->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer2/block1/sub1/sub1_conv/Conv2D
        Conv2D_6 = new Convolution("Conv2D_6", 64,
                        BatchNorm_2->getOutputChannels(), BatchNorm_2->getOutputHeight(), BatchNorm_2->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block1/sub2/sub2_conv/Conv2D
        Conv2D_7 = new Convolution("Conv2D_7", 64,
                        Conv2D_6->getOutputChannels(), Conv2D_6->getOutputHeight(), Conv2D_6->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block1/sub3/sub3_conv/Conv2D
        Conv2D_8 = new Convolution("Conv2D_8", 256,
                        Conv2D_7->getOutputChannels(), Conv2D_7->getOutputHeight(), Conv2D_7->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer2/block1/shortcut/add
        Add_2 = new Sum("Add_2", 
                        Conv2D_8->getOutputChannels(), Conv2D_8->getOutputHeight(), Conv2D_8->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block0/common_bn_relu/FusedBatchNorm
        BatchNorm_3 = new BatchNorm("BatchNorm_3", 
                        Add_2->getOutputChannels(), Add_2->getOutputHeight(), Add_2->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block0/sub1/sub1_conv/Conv2D
        Conv2D_9 = new Convolution("Conv2D_9", 128,
                        BatchNorm_3->getOutputChannels(), BatchNorm_3->getOutputHeight(), BatchNorm_3->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block0/sub2/sub2_conv/Conv2D
        Conv2D_10 = new Convolution("Conv2D_10", 128,
                        Conv2D_9->getOutputChannels(), Conv2D_9->getOutputHeight(), Conv2D_9->getOutputWidth(),
                        3, 3, 0, 1, 0, 1, 2, 2,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block0/sub3/sub3_conv/Conv2D
        Conv2D_11 = new Convolution("Conv2D_11", 512,
                        Conv2D_10->getOutputChannels(), Conv2D_10->getOutputHeight(), Conv2D_10->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block0/shortcut/sub_sc/Conv2D
        Conv2D_12 = new Convolution("Conv2D_12", 512,
                        BatchNorm_3->getOutputChannels(), BatchNorm_3->getOutputHeight(), BatchNorm_3->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 2, 2,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block0/shortcut/add
        Add_3 = new Sum("Add_3", 
                        Conv2D_11->getOutputChannels(), Conv2D_11->getOutputHeight(), Conv2D_11->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block1/residual_bn_relu/FusedBatchNorm
        BatchNorm_4 = new BatchNorm("BatchNorm_4", 
                        Add_3->getOutputChannels(), Add_3->getOutputHeight(), Add_3->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block1/sub1/sub1_conv/Conv2D
        Conv2D_13 = new Convolution("Conv2D_13", 128,
                        BatchNorm_4->getOutputChannels(), BatchNorm_4->getOutputHeight(), BatchNorm_4->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block1/sub2/sub2_conv/Conv2D
        Conv2D_14 = new Convolution("Conv2D_14", 128,
                        Conv2D_13->getOutputChannels(), Conv2D_13->getOutputHeight(), Conv2D_13->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block1/sub3/sub3_conv/Conv2D
        Conv2D_15 = new Convolution("Conv2D_15", 512,
                        Conv2D_14->getOutputChannels(), Conv2D_14->getOutputHeight(), Conv2D_14->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block1/shortcut/add
        Add_4 = new Sum("Add_4", 
                        Conv2D_15->getOutputChannels(), Conv2D_15->getOutputHeight(), Conv2D_15->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block2/residual_bn_relu/FusedBatchNorm
        BatchNorm_5 = new BatchNorm("BatchNorm_5", 
                        Add_4->getOutputChannels(), Add_4->getOutputHeight(), Add_4->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block2/sub1/sub1_conv/Conv2D
        Conv2D_16 = new Convolution("Conv2D_16", 128,
                        BatchNorm_5->getOutputChannels(), BatchNorm_5->getOutputHeight(), BatchNorm_5->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block2/sub2/sub2_conv/Conv2D
        Conv2D_17 = new Convolution("Conv2D_17", 128,
                        Conv2D_16->getOutputChannels(), Conv2D_16->getOutputHeight(), Conv2D_16->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block2/sub3/sub3_conv/Conv2D
        Conv2D_18 = new Convolution("Conv2D_18", 512,
                        Conv2D_17->getOutputChannels(), Conv2D_17->getOutputHeight(), Conv2D_17->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block2/shortcut/add
        Add_5 = new Sum("Add_5", 
                        Conv2D_18->getOutputChannels(), Conv2D_18->getOutputHeight(), Conv2D_18->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block3/residual_bn_relu/FusedBatchNorm
        BatchNorm_6 = new BatchNorm("BatchNorm_6", 
                        Add_5->getOutputChannels(), Add_5->getOutputHeight(), Add_5->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block3/sub1/sub1_conv/Conv2D
        Conv2D_19 = new Convolution("Conv2D_19", 128,
                        BatchNorm_6->getOutputChannels(), BatchNorm_6->getOutputHeight(), BatchNorm_6->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block3/sub2/sub2_conv/Conv2D
        Conv2D_20 = new Convolution("Conv2D_20", 128,
                        Conv2D_19->getOutputChannels(), Conv2D_19->getOutputHeight(), Conv2D_19->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block3/sub3/sub3_conv/Conv2D
        Conv2D_21 = new Convolution("Conv2D_21", 512,
                        Conv2D_20->getOutputChannels(), Conv2D_20->getOutputHeight(), Conv2D_20->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block3/shortcut/add
        Add_6 = new Sum("Add_6", 
                        Conv2D_21->getOutputChannels(), Conv2D_21->getOutputHeight(), Conv2D_21->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block4/residual_bn_relu/FusedBatchNorm
        BatchNorm_7 = new BatchNorm("BatchNorm_7", 
                        Add_6->getOutputChannels(), Add_6->getOutputHeight(), Add_6->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block4/sub1/sub1_conv/Conv2D
        Conv2D_22 = new Convolution("Conv2D_22", 128,
                        BatchNorm_7->getOutputChannels(), BatchNorm_7->getOutputHeight(), BatchNorm_7->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block4/sub2/sub2_conv/Conv2D
        Conv2D_23 = new Convolution("Conv2D_23", 128,
                        Conv2D_22->getOutputChannels(), Conv2D_22->getOutputHeight(), Conv2D_22->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block4/sub3/sub3_conv/Conv2D
        Conv2D_24 = new Convolution("Conv2D_24", 512,
                        Conv2D_23->getOutputChannels(), Conv2D_23->getOutputHeight(), Conv2D_23->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block4/shortcut/add
        Add_7 = new Sum("Add_7", 
                        Conv2D_24->getOutputChannels(), Conv2D_24->getOutputHeight(), Conv2D_24->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block5/residual_bn_relu/FusedBatchNorm
        BatchNorm_8 = new BatchNorm("BatchNorm_8", 
                        Add_7->getOutputChannels(), Add_7->getOutputHeight(), Add_7->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block5/sub1/sub1_conv/Conv2D
        Conv2D_25 = new Convolution("Conv2D_25", 128,
                        BatchNorm_8->getOutputChannels(), BatchNorm_8->getOutputHeight(), BatchNorm_8->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block5/sub2/sub2_conv/Conv2D
        Conv2D_26 = new Convolution("Conv2D_26", 128,
                        Conv2D_25->getOutputChannels(), Conv2D_25->getOutputHeight(), Conv2D_25->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block5/sub3/sub3_conv/Conv2D
        Conv2D_27 = new Convolution("Conv2D_27", 512,
                        Conv2D_26->getOutputChannels(), Conv2D_26->getOutputHeight(), Conv2D_26->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block5/shortcut/add
        Add_8 = new Sum("Add_8", 
                        Conv2D_27->getOutputChannels(), Conv2D_27->getOutputHeight(), Conv2D_27->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer3/block6/residual_bn_relu/FusedBatchNorm
        BatchNorm_9 = new BatchNorm("BatchNorm_9", 
                        Add_8->getOutputChannels(), Add_8->getOutputHeight(), Add_8->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer3/block6/sub1/sub1_conv/Conv2D
        Conv2D_28 = new Convolution("Conv2D_28", 128,
                        BatchNorm_9->getOutputChannels(), BatchNorm_9->getOutputHeight(), BatchNorm_9->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block6/sub2/sub2_conv/Conv2D
        Conv2D_29 = new Convolution("Conv2D_29", 128,
                        Conv2D_28->getOutputChannels(), Conv2D_28->getOutputHeight(), Conv2D_28->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block6/sub3/sub3_conv/Conv2D
        Conv2D_30 = new Convolution("Conv2D_30", 512,
                        Conv2D_29->getOutputChannels(), Conv2D_29->getOutputHeight(), Conv2D_29->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer3/block6/shortcut/add
        Add_9 = new Sum("Add_9", 
                        Conv2D_30->getOutputChannels(), Conv2D_30->getOutputHeight(), Conv2D_30->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block0/common_bn_relu/FusedBatchNorm
        BatchNorm_10 = new BatchNorm("BatchNorm_10", 
                        Add_9->getOutputChannels(), Add_9->getOutputHeight(), Add_9->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block0/sub1/sub1_conv/Conv2D
        Conv2D_31 = new Convolution("Conv2D_31", 256,
                        BatchNorm_10->getOutputChannels(), BatchNorm_10->getOutputHeight(), BatchNorm_10->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block0/sub2/sub2_conv/Conv2D
        Conv2D_32 = new Convolution("Conv2D_32", 256,
                        Conv2D_31->getOutputChannels(), Conv2D_31->getOutputHeight(), Conv2D_31->getOutputWidth(),
                        3, 3, 0, 1, 0, 1, 2, 2,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block0/sub3/sub3_conv/Conv2D
        Conv2D_33 = new Convolution("Conv2D_33", 1024,
                        Conv2D_32->getOutputChannels(), Conv2D_32->getOutputHeight(), Conv2D_32->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block0/shortcut/sub_sc/Conv2D
        Conv2D_34 = new Convolution("Conv2D_34", 1024,
                        BatchNorm_10->getOutputChannels(), BatchNorm_10->getOutputHeight(), BatchNorm_10->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 2, 2,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block0/shortcut/add
        Add_10 = new Sum("Add_10", 
                        Conv2D_33->getOutputChannels(), Conv2D_33->getOutputHeight(), Conv2D_33->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block1/residual_bn_relu/FusedBatchNorm
        BatchNorm_11 = new BatchNorm("BatchNorm_11", 
                        Add_10->getOutputChannels(), Add_10->getOutputHeight(), Add_10->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block1/sub1/sub1_conv/Conv2D
        Conv2D_35 = new Convolution("Conv2D_35", 256,
                        BatchNorm_11->getOutputChannels(), BatchNorm_11->getOutputHeight(), BatchNorm_11->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block1/sub2/sub2_conv/Conv2D
        Conv2D_36 = new Convolution("Conv2D_36", 256,
                        Conv2D_35->getOutputChannels(), Conv2D_35->getOutputHeight(), Conv2D_35->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block1/sub3/sub3_conv/Conv2D
        Conv2D_37 = new Convolution("Conv2D_37", 1024,
                        Conv2D_36->getOutputChannels(), Conv2D_36->getOutputHeight(), Conv2D_36->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block1/shortcut/add
        Add_11 = new Sum("Add_11", 
                        Conv2D_37->getOutputChannels(), Conv2D_37->getOutputHeight(), Conv2D_37->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block2/residual_bn_relu/FusedBatchNorm
        BatchNorm_12 = new BatchNorm("BatchNorm_12", 
                        Add_11->getOutputChannels(), Add_11->getOutputHeight(), Add_11->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block2/sub1/sub1_conv/Conv2D
        Conv2D_38 = new Convolution("Conv2D_38", 256,
                        BatchNorm_12->getOutputChannels(), BatchNorm_12->getOutputHeight(), BatchNorm_12->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block2/sub2/sub2_conv/Conv2D
        Conv2D_39 = new Convolution("Conv2D_39", 256,
                        Conv2D_38->getOutputChannels(), Conv2D_38->getOutputHeight(), Conv2D_38->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block2/sub3/sub3_conv/Conv2D
        Conv2D_40 = new Convolution("Conv2D_40", 1024,
                        Conv2D_39->getOutputChannels(), Conv2D_39->getOutputHeight(), Conv2D_39->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block2/shortcut/add
        Add_12 = new Sum("Add_12", 
                        Conv2D_40->getOutputChannels(), Conv2D_40->getOutputHeight(), Conv2D_40->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block3/residual_bn_relu/FusedBatchNorm
        BatchNorm_13 = new BatchNorm("BatchNorm_13", 
                        Add_12->getOutputChannels(), Add_12->getOutputHeight(), Add_12->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block3/sub1/sub1_conv/Conv2D
        Conv2D_41 = new Convolution("Conv2D_41", 256,
                        BatchNorm_13->getOutputChannels(), BatchNorm_13->getOutputHeight(), BatchNorm_13->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block3/sub2/sub2_conv/Conv2D
        Conv2D_42 = new Convolution("Conv2D_42", 256,
                        Conv2D_41->getOutputChannels(), Conv2D_41->getOutputHeight(), Conv2D_41->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block3/sub3/sub3_conv/Conv2D
        Conv2D_43 = new Convolution("Conv2D_43", 1024,
                        Conv2D_42->getOutputChannels(), Conv2D_42->getOutputHeight(), Conv2D_42->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block3/shortcut/add
        Add_13 = new Sum("Add_13", 
                        Conv2D_43->getOutputChannels(), Conv2D_43->getOutputHeight(), Conv2D_43->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block4/residual_bn_relu/FusedBatchNorm
        BatchNorm_14 = new BatchNorm("BatchNorm_14", 
                        Add_13->getOutputChannels(), Add_13->getOutputHeight(), Add_13->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block4/sub1/sub1_conv/Conv2D
        Conv2D_44 = new Convolution("Conv2D_44", 256,
                        BatchNorm_14->getOutputChannels(), BatchNorm_14->getOutputHeight(), BatchNorm_14->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block4/sub2/sub2_conv/Conv2D
        Conv2D_45 = new Convolution("Conv2D_45", 256,
                        Conv2D_44->getOutputChannels(), Conv2D_44->getOutputHeight(), Conv2D_44->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block4/sub3/sub3_conv/Conv2D
        Conv2D_46 = new Convolution("Conv2D_46", 1024,
                        Conv2D_45->getOutputChannels(), Conv2D_45->getOutputHeight(), Conv2D_45->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block4/shortcut/add
        Add_14 = new Sum("Add_14", 
                        Conv2D_46->getOutputChannels(), Conv2D_46->getOutputHeight(), Conv2D_46->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block5/residual_bn_relu/FusedBatchNorm
        BatchNorm_15 = new BatchNorm("BatchNorm_15", 
                        Add_14->getOutputChannels(), Add_14->getOutputHeight(), Add_14->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block5/sub1/sub1_conv/Conv2D
        Conv2D_47 = new Convolution("Conv2D_47", 256,
                        BatchNorm_15->getOutputChannels(), BatchNorm_15->getOutputHeight(), BatchNorm_15->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block5/sub2/sub2_conv/Conv2D
        Conv2D_48 = new Convolution("Conv2D_48", 256,
                        Conv2D_47->getOutputChannels(), Conv2D_47->getOutputHeight(), Conv2D_47->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block5/sub3/sub3_conv/Conv2D
        Conv2D_49 = new Convolution("Conv2D_49", 1024,
                        Conv2D_48->getOutputChannels(), Conv2D_48->getOutputHeight(), Conv2D_48->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block5/shortcut/add
        Add_15 = new Sum("Add_15", 
                        Conv2D_49->getOutputChannels(), Conv2D_49->getOutputHeight(), Conv2D_49->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block6/residual_bn_relu/FusedBatchNorm
        BatchNorm_16 = new BatchNorm("BatchNorm_16", 
                        Add_15->getOutputChannels(), Add_15->getOutputHeight(), Add_15->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block6/sub1/sub1_conv/Conv2D
        Conv2D_50 = new Convolution("Conv2D_50", 256,
                        BatchNorm_16->getOutputChannels(), BatchNorm_16->getOutputHeight(), BatchNorm_16->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block6/sub2/sub2_conv/Conv2D
        Conv2D_51 = new Convolution("Conv2D_51", 256,
                        Conv2D_50->getOutputChannels(), Conv2D_50->getOutputHeight(), Conv2D_50->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block6/sub3/sub3_conv/Conv2D
        Conv2D_52 = new Convolution("Conv2D_52", 1024,
                        Conv2D_51->getOutputChannels(), Conv2D_51->getOutputHeight(), Conv2D_51->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block6/shortcut/add
        Add_16 = new Sum("Add_16", 
                        Conv2D_52->getOutputChannels(), Conv2D_52->getOutputHeight(), Conv2D_52->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block7/residual_bn_relu/FusedBatchNorm
        BatchNorm_17 = new BatchNorm("BatchNorm_17", 
                        Add_16->getOutputChannels(), Add_16->getOutputHeight(), Add_16->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block7/sub1/sub1_conv/Conv2D
        Conv2D_53 = new Convolution("Conv2D_53", 256,
                        BatchNorm_17->getOutputChannels(), BatchNorm_17->getOutputHeight(), BatchNorm_17->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block7/sub2/sub2_conv/Conv2D
        Conv2D_54 = new Convolution("Conv2D_54", 256,
                        Conv2D_53->getOutputChannels(), Conv2D_53->getOutputHeight(), Conv2D_53->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block7/sub3/sub3_conv/Conv2D
        Conv2D_55 = new Convolution("Conv2D_55", 1024,
                        Conv2D_54->getOutputChannels(), Conv2D_54->getOutputHeight(), Conv2D_54->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block7/shortcut/add
        Add_17 = new Sum("Add_17", 
                        Conv2D_55->getOutputChannels(), Conv2D_55->getOutputHeight(), Conv2D_55->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block8/residual_bn_relu/FusedBatchNorm
        BatchNorm_18 = new BatchNorm("BatchNorm_18", 
                        Add_17->getOutputChannels(), Add_17->getOutputHeight(), Add_17->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block8/sub1/sub1_conv/Conv2D
        Conv2D_56 = new Convolution("Conv2D_56", 256,
                        BatchNorm_18->getOutputChannels(), BatchNorm_18->getOutputHeight(), BatchNorm_18->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block8/sub2/sub2_conv/Conv2D
        Conv2D_57 = new Convolution("Conv2D_57", 256,
                        Conv2D_56->getOutputChannels(), Conv2D_56->getOutputHeight(), Conv2D_56->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block8/sub3/sub3_conv/Conv2D
        Conv2D_58 = new Convolution("Conv2D_58", 1024,
                        Conv2D_57->getOutputChannels(), Conv2D_57->getOutputHeight(), Conv2D_57->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block8/shortcut/add
        Add_18 = new Sum("Add_18", 
                        Conv2D_58->getOutputChannels(), Conv2D_58->getOutputHeight(), Conv2D_58->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block9/residual_bn_relu/FusedBatchNorm
        BatchNorm_19 = new BatchNorm("BatchNorm_19", 
                        Add_18->getOutputChannels(), Add_18->getOutputHeight(), Add_18->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block9/sub1/sub1_conv/Conv2D
        Conv2D_59 = new Convolution("Conv2D_59", 256,
                        BatchNorm_19->getOutputChannels(), BatchNorm_19->getOutputHeight(), BatchNorm_19->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block9/sub2/sub2_conv/Conv2D
        Conv2D_60 = new Convolution("Conv2D_60", 256,
                        Conv2D_59->getOutputChannels(), Conv2D_59->getOutputHeight(), Conv2D_59->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block9/sub3/sub3_conv/Conv2D
        Conv2D_61 = new Convolution("Conv2D_61", 1024,
                        Conv2D_60->getOutputChannels(), Conv2D_60->getOutputHeight(), Conv2D_60->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block9/shortcut/add
        Add_19 = new Sum("Add_19", 
                        Conv2D_61->getOutputChannels(), Conv2D_61->getOutputHeight(), Conv2D_61->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block10/residual_bn_relu/FusedBatchNorm
        BatchNorm_20 = new BatchNorm("BatchNorm_20", 
                        Add_19->getOutputChannels(), Add_19->getOutputHeight(), Add_19->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block10/sub1/sub1_conv/Conv2D
        Conv2D_62 = new Convolution("Conv2D_62", 256,
                        BatchNorm_20->getOutputChannels(), BatchNorm_20->getOutputHeight(), BatchNorm_20->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block10/sub2/sub2_conv/Conv2D
        Conv2D_63 = new Convolution("Conv2D_63", 256,
                        Conv2D_62->getOutputChannels(), Conv2D_62->getOutputHeight(), Conv2D_62->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block10/sub3/sub3_conv/Conv2D
        Conv2D_64 = new Convolution("Conv2D_64", 1024,
                        Conv2D_63->getOutputChannels(), Conv2D_63->getOutputHeight(), Conv2D_63->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block10/shortcut/add
        Add_20 = new Sum("Add_20", 
                        Conv2D_64->getOutputChannels(), Conv2D_64->getOutputHeight(), Conv2D_64->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block11/residual_bn_relu/FusedBatchNorm
        BatchNorm_21 = new BatchNorm("BatchNorm_21", 
                        Add_20->getOutputChannels(), Add_20->getOutputHeight(), Add_20->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block11/sub1/sub1_conv/Conv2D
        Conv2D_65 = new Convolution("Conv2D_65", 256,
                        BatchNorm_21->getOutputChannels(), BatchNorm_21->getOutputHeight(), BatchNorm_21->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block11/sub2/sub2_conv/Conv2D
        Conv2D_66 = new Convolution("Conv2D_66", 256,
                        Conv2D_65->getOutputChannels(), Conv2D_65->getOutputHeight(), Conv2D_65->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block11/sub3/sub3_conv/Conv2D
        Conv2D_67 = new Convolution("Conv2D_67", 1024,
                        Conv2D_66->getOutputChannels(), Conv2D_66->getOutputHeight(), Conv2D_66->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block11/shortcut/add
        Add_21 = new Sum("Add_21", 
                        Conv2D_67->getOutputChannels(), Conv2D_67->getOutputHeight(), Conv2D_67->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block12/residual_bn_relu/FusedBatchNorm
        BatchNorm_22 = new BatchNorm("BatchNorm_22", 
                        Add_21->getOutputChannels(), Add_21->getOutputHeight(), Add_21->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block12/sub1/sub1_conv/Conv2D
        Conv2D_68 = new Convolution("Conv2D_68", 256,
                        BatchNorm_22->getOutputChannels(), BatchNorm_22->getOutputHeight(), BatchNorm_22->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block12/sub2/sub2_conv/Conv2D
        Conv2D_69 = new Convolution("Conv2D_69", 256,
                        Conv2D_68->getOutputChannels(), Conv2D_68->getOutputHeight(), Conv2D_68->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block12/sub3/sub3_conv/Conv2D
        Conv2D_70 = new Convolution("Conv2D_70", 1024,
                        Conv2D_69->getOutputChannels(), Conv2D_69->getOutputHeight(), Conv2D_69->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block12/shortcut/add
        Add_22 = new Sum("Add_22", 
                        Conv2D_70->getOutputChannels(), Conv2D_70->getOutputHeight(), Conv2D_70->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block13/residual_bn_relu/FusedBatchNorm
        BatchNorm_23 = new BatchNorm("BatchNorm_23", 
                        Add_22->getOutputChannels(), Add_22->getOutputHeight(), Add_22->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block13/sub1/sub1_conv/Conv2D
        Conv2D_71 = new Convolution("Conv2D_71", 256,
                        BatchNorm_23->getOutputChannels(), BatchNorm_23->getOutputHeight(), BatchNorm_23->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block13/sub2/sub2_conv/Conv2D
        Conv2D_72 = new Convolution("Conv2D_72", 256,
                        Conv2D_71->getOutputChannels(), Conv2D_71->getOutputHeight(), Conv2D_71->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block13/sub3/sub3_conv/Conv2D
        Conv2D_73 = new Convolution("Conv2D_73", 1024,
                        Conv2D_72->getOutputChannels(), Conv2D_72->getOutputHeight(), Conv2D_72->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block13/shortcut/add
        Add_23 = new Sum("Add_23", 
                        Conv2D_73->getOutputChannels(), Conv2D_73->getOutputHeight(), Conv2D_73->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block14/residual_bn_relu/FusedBatchNorm
        BatchNorm_24 = new BatchNorm("BatchNorm_24", 
                        Add_23->getOutputChannels(), Add_23->getOutputHeight(), Add_23->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block14/sub1/sub1_conv/Conv2D
        Conv2D_74 = new Convolution("Conv2D_74", 256,
                        BatchNorm_24->getOutputChannels(), BatchNorm_24->getOutputHeight(), BatchNorm_24->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block14/sub2/sub2_conv/Conv2D
        Conv2D_75 = new Convolution("Conv2D_75", 256,
                        Conv2D_74->getOutputChannels(), Conv2D_74->getOutputHeight(), Conv2D_74->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block14/sub3/sub3_conv/Conv2D
        Conv2D_76 = new Convolution("Conv2D_76", 1024,
                        Conv2D_75->getOutputChannels(), Conv2D_75->getOutputHeight(), Conv2D_75->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block14/shortcut/add
        Add_24 = new Sum("Add_24", 
                        Conv2D_76->getOutputChannels(), Conv2D_76->getOutputHeight(), Conv2D_76->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block15/residual_bn_relu/FusedBatchNorm
        BatchNorm_25 = new BatchNorm("BatchNorm_25", 
                        Add_24->getOutputChannels(), Add_24->getOutputHeight(), Add_24->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block15/sub1/sub1_conv/Conv2D
        Conv2D_77 = new Convolution("Conv2D_77", 256,
                        BatchNorm_25->getOutputChannels(), BatchNorm_25->getOutputHeight(), BatchNorm_25->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block15/sub2/sub2_conv/Conv2D
        Conv2D_78 = new Convolution("Conv2D_78", 256,
                        Conv2D_77->getOutputChannels(), Conv2D_77->getOutputHeight(), Conv2D_77->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block15/sub3/sub3_conv/Conv2D
        Conv2D_79 = new Convolution("Conv2D_79", 1024,
                        Conv2D_78->getOutputChannels(), Conv2D_78->getOutputHeight(), Conv2D_78->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block15/shortcut/add
        Add_25 = new Sum("Add_25", 
                        Conv2D_79->getOutputChannels(), Conv2D_79->getOutputHeight(), Conv2D_79->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block16/residual_bn_relu/FusedBatchNorm
        BatchNorm_26 = new BatchNorm("BatchNorm_26", 
                        Add_25->getOutputChannels(), Add_25->getOutputHeight(), Add_25->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block16/sub1/sub1_conv/Conv2D
        Conv2D_80 = new Convolution("Conv2D_80", 256,
                        BatchNorm_26->getOutputChannels(), BatchNorm_26->getOutputHeight(), BatchNorm_26->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block16/sub2/sub2_conv/Conv2D
        Conv2D_81 = new Convolution("Conv2D_81", 256,
                        Conv2D_80->getOutputChannels(), Conv2D_80->getOutputHeight(), Conv2D_80->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block16/sub3/sub3_conv/Conv2D
        Conv2D_82 = new Convolution("Conv2D_82", 1024,
                        Conv2D_81->getOutputChannels(), Conv2D_81->getOutputHeight(), Conv2D_81->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block16/shortcut/add
        Add_26 = new Sum("Add_26", 
                        Conv2D_82->getOutputChannels(), Conv2D_82->getOutputHeight(), Conv2D_82->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block17/residual_bn_relu/FusedBatchNorm
        BatchNorm_27 = new BatchNorm("BatchNorm_27", 
                        Add_26->getOutputChannels(), Add_26->getOutputHeight(), Add_26->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block17/sub1/sub1_conv/Conv2D
        Conv2D_83 = new Convolution("Conv2D_83", 256,
                        BatchNorm_27->getOutputChannels(), BatchNorm_27->getOutputHeight(), BatchNorm_27->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block17/sub2/sub2_conv/Conv2D
        Conv2D_84 = new Convolution("Conv2D_84", 256,
                        Conv2D_83->getOutputChannels(), Conv2D_83->getOutputHeight(), Conv2D_83->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block17/sub3/sub3_conv/Conv2D
        Conv2D_85 = new Convolution("Conv2D_85", 1024,
                        Conv2D_84->getOutputChannels(), Conv2D_84->getOutputHeight(), Conv2D_84->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block17/shortcut/add
        Add_27 = new Sum("Add_27", 
                        Conv2D_85->getOutputChannels(), Conv2D_85->getOutputHeight(), Conv2D_85->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block18/residual_bn_relu/FusedBatchNorm
        BatchNorm_28 = new BatchNorm("BatchNorm_28", 
                        Add_27->getOutputChannels(), Add_27->getOutputHeight(), Add_27->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block18/sub1/sub1_conv/Conv2D
        Conv2D_86 = new Convolution("Conv2D_86", 256,
                        BatchNorm_28->getOutputChannels(), BatchNorm_28->getOutputHeight(), BatchNorm_28->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block18/sub2/sub2_conv/Conv2D
        Conv2D_87 = new Convolution("Conv2D_87", 256,
                        Conv2D_86->getOutputChannels(), Conv2D_86->getOutputHeight(), Conv2D_86->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block18/sub3/sub3_conv/Conv2D
        Conv2D_88 = new Convolution("Conv2D_88", 1024,
                        Conv2D_87->getOutputChannels(), Conv2D_87->getOutputHeight(), Conv2D_87->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block18/shortcut/add
        Add_28 = new Sum("Add_28", 
                        Conv2D_88->getOutputChannels(), Conv2D_88->getOutputHeight(), Conv2D_88->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block19/residual_bn_relu/FusedBatchNorm
        BatchNorm_29 = new BatchNorm("BatchNorm_29", 
                        Add_28->getOutputChannels(), Add_28->getOutputHeight(), Add_28->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block19/sub1/sub1_conv/Conv2D
        Conv2D_89 = new Convolution("Conv2D_89", 256,
                        BatchNorm_29->getOutputChannels(), BatchNorm_29->getOutputHeight(), BatchNorm_29->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block19/sub2/sub2_conv/Conv2D
        Conv2D_90 = new Convolution("Conv2D_90", 256,
                        Conv2D_89->getOutputChannels(), Conv2D_89->getOutputHeight(), Conv2D_89->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block19/sub3/sub3_conv/Conv2D
        Conv2D_91 = new Convolution("Conv2D_91", 1024,
                        Conv2D_90->getOutputChannels(), Conv2D_90->getOutputHeight(), Conv2D_90->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block19/shortcut/add
        Add_29 = new Sum("Add_29", 
                        Conv2D_91->getOutputChannels(), Conv2D_91->getOutputHeight(), Conv2D_91->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block20/residual_bn_relu/FusedBatchNorm
        BatchNorm_30 = new BatchNorm("BatchNorm_30", 
                        Add_29->getOutputChannels(), Add_29->getOutputHeight(), Add_29->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block20/sub1/sub1_conv/Conv2D
        Conv2D_92 = new Convolution("Conv2D_92", 256,
                        BatchNorm_30->getOutputChannels(), BatchNorm_30->getOutputHeight(), BatchNorm_30->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block20/sub2/sub2_conv/Conv2D
        Conv2D_93 = new Convolution("Conv2D_93", 256,
                        Conv2D_92->getOutputChannels(), Conv2D_92->getOutputHeight(), Conv2D_92->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block20/sub3/sub3_conv/Conv2D
        Conv2D_94 = new Convolution("Conv2D_94", 1024,
                        Conv2D_93->getOutputChannels(), Conv2D_93->getOutputHeight(), Conv2D_93->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block20/shortcut/add
        Add_30 = new Sum("Add_30", 
                        Conv2D_94->getOutputChannels(), Conv2D_94->getOutputHeight(), Conv2D_94->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block21/residual_bn_relu/FusedBatchNorm
        BatchNorm_31 = new BatchNorm("BatchNorm_31", 
                        Add_30->getOutputChannels(), Add_30->getOutputHeight(), Add_30->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block21/sub1/sub1_conv/Conv2D
        Conv2D_95 = new Convolution("Conv2D_95", 256,
                        BatchNorm_31->getOutputChannels(), BatchNorm_31->getOutputHeight(), BatchNorm_31->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block21/sub2/sub2_conv/Conv2D
        Conv2D_96 = new Convolution("Conv2D_96", 256,
                        Conv2D_95->getOutputChannels(), Conv2D_95->getOutputHeight(), Conv2D_95->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block21/sub3/sub3_conv/Conv2D
        Conv2D_97 = new Convolution("Conv2D_97", 1024,
                        Conv2D_96->getOutputChannels(), Conv2D_96->getOutputHeight(), Conv2D_96->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block21/shortcut/add
        Add_31 = new Sum("Add_31", 
                        Conv2D_97->getOutputChannels(), Conv2D_97->getOutputHeight(), Conv2D_97->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block22/residual_bn_relu/FusedBatchNorm
        BatchNorm_32 = new BatchNorm("BatchNorm_32", 
                        Add_31->getOutputChannels(), Add_31->getOutputHeight(), Add_31->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block22/sub1/sub1_conv/Conv2D
        Conv2D_98 = new Convolution("Conv2D_98", 256,
                        BatchNorm_32->getOutputChannels(), BatchNorm_32->getOutputHeight(), BatchNorm_32->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block22/sub2/sub2_conv/Conv2D
        Conv2D_99 = new Convolution("Conv2D_99", 256,
                        Conv2D_98->getOutputChannels(), Conv2D_98->getOutputHeight(), Conv2D_98->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block22/sub3/sub3_conv/Conv2D
        Conv2D_100 = new Convolution("Conv2D_100", 1024,
                        Conv2D_99->getOutputChannels(), Conv2D_99->getOutputHeight(), Conv2D_99->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block22/shortcut/add
        Add_32 = new Sum("Add_32", 
                        Conv2D_100->getOutputChannels(), Conv2D_100->getOutputHeight(), Conv2D_100->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block23/residual_bn_relu/FusedBatchNorm
        BatchNorm_33 = new BatchNorm("BatchNorm_33", 
                        Add_32->getOutputChannels(), Add_32->getOutputHeight(), Add_32->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block23/sub1/sub1_conv/Conv2D
        Conv2D_101 = new Convolution("Conv2D_101", 256,
                        BatchNorm_33->getOutputChannels(), BatchNorm_33->getOutputHeight(), BatchNorm_33->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block23/sub2/sub2_conv/Conv2D
        Conv2D_102 = new Convolution("Conv2D_102", 256,
                        Conv2D_101->getOutputChannels(), Conv2D_101->getOutputHeight(), Conv2D_101->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block23/sub3/sub3_conv/Conv2D
        Conv2D_103 = new Convolution("Conv2D_103", 1024,
                        Conv2D_102->getOutputChannels(), Conv2D_102->getOutputHeight(), Conv2D_102->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block23/shortcut/add
        Add_33 = new Sum("Add_33", 
                        Conv2D_103->getOutputChannels(), Conv2D_103->getOutputHeight(), Conv2D_103->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block24/residual_bn_relu/FusedBatchNorm
        BatchNorm_34 = new BatchNorm("BatchNorm_34", 
                        Add_33->getOutputChannels(), Add_33->getOutputHeight(), Add_33->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block24/sub1/sub1_conv/Conv2D
        Conv2D_104 = new Convolution("Conv2D_104", 256,
                        BatchNorm_34->getOutputChannels(), BatchNorm_34->getOutputHeight(), BatchNorm_34->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block24/sub2/sub2_conv/Conv2D
        Conv2D_105 = new Convolution("Conv2D_105", 256,
                        Conv2D_104->getOutputChannels(), Conv2D_104->getOutputHeight(), Conv2D_104->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block24/sub3/sub3_conv/Conv2D
        Conv2D_106 = new Convolution("Conv2D_106", 1024,
                        Conv2D_105->getOutputChannels(), Conv2D_105->getOutputHeight(), Conv2D_105->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block24/shortcut/add
        Add_34 = new Sum("Add_34", 
                        Conv2D_106->getOutputChannels(), Conv2D_106->getOutputHeight(), Conv2D_106->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block25/residual_bn_relu/FusedBatchNorm
        BatchNorm_35 = new BatchNorm("BatchNorm_35", 
                        Add_34->getOutputChannels(), Add_34->getOutputHeight(), Add_34->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block25/sub1/sub1_conv/Conv2D
        Conv2D_107 = new Convolution("Conv2D_107", 256,
                        BatchNorm_35->getOutputChannels(), BatchNorm_35->getOutputHeight(), BatchNorm_35->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block25/sub2/sub2_conv/Conv2D
        Conv2D_108 = new Convolution("Conv2D_108", 256,
                        Conv2D_107->getOutputChannels(), Conv2D_107->getOutputHeight(), Conv2D_107->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block25/sub3/sub3_conv/Conv2D
        Conv2D_109 = new Convolution("Conv2D_109", 1024,
                        Conv2D_108->getOutputChannels(), Conv2D_108->getOutputHeight(), Conv2D_108->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block25/shortcut/add
        Add_35 = new Sum("Add_35", 
                        Conv2D_109->getOutputChannels(), Conv2D_109->getOutputHeight(), Conv2D_109->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block26/residual_bn_relu/FusedBatchNorm
        BatchNorm_36 = new BatchNorm("BatchNorm_36", 
                        Add_35->getOutputChannels(), Add_35->getOutputHeight(), Add_35->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block26/sub1/sub1_conv/Conv2D
        Conv2D_110 = new Convolution("Conv2D_110", 256,
                        BatchNorm_36->getOutputChannels(), BatchNorm_36->getOutputHeight(), BatchNorm_36->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block26/sub2/sub2_conv/Conv2D
        Conv2D_111 = new Convolution("Conv2D_111", 256,
                        Conv2D_110->getOutputChannels(), Conv2D_110->getOutputHeight(), Conv2D_110->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block26/sub3/sub3_conv/Conv2D
        Conv2D_112 = new Convolution("Conv2D_112", 1024,
                        Conv2D_111->getOutputChannels(), Conv2D_111->getOutputHeight(), Conv2D_111->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block26/shortcut/add
        Add_36 = new Sum("Add_36", 
                        Conv2D_112->getOutputChannels(), Conv2D_112->getOutputHeight(), Conv2D_112->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block27/residual_bn_relu/FusedBatchNorm
        BatchNorm_37 = new BatchNorm("BatchNorm_37", 
                        Add_36->getOutputChannels(), Add_36->getOutputHeight(), Add_36->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block27/sub1/sub1_conv/Conv2D
        Conv2D_113 = new Convolution("Conv2D_113", 256,
                        BatchNorm_37->getOutputChannels(), BatchNorm_37->getOutputHeight(), BatchNorm_37->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block27/sub2/sub2_conv/Conv2D
        Conv2D_114 = new Convolution("Conv2D_114", 256,
                        Conv2D_113->getOutputChannels(), Conv2D_113->getOutputHeight(), Conv2D_113->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block27/sub3/sub3_conv/Conv2D
        Conv2D_115 = new Convolution("Conv2D_115", 1024,
                        Conv2D_114->getOutputChannels(), Conv2D_114->getOutputHeight(), Conv2D_114->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block27/shortcut/add
        Add_37 = new Sum("Add_37", 
                        Conv2D_115->getOutputChannels(), Conv2D_115->getOutputHeight(), Conv2D_115->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block28/residual_bn_relu/FusedBatchNorm
        BatchNorm_38 = new BatchNorm("BatchNorm_38", 
                        Add_37->getOutputChannels(), Add_37->getOutputHeight(), Add_37->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block28/sub1/sub1_conv/Conv2D
        Conv2D_116 = new Convolution("Conv2D_116", 256,
                        BatchNorm_38->getOutputChannels(), BatchNorm_38->getOutputHeight(), BatchNorm_38->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block28/sub2/sub2_conv/Conv2D
        Conv2D_117 = new Convolution("Conv2D_117", 256,
                        Conv2D_116->getOutputChannels(), Conv2D_116->getOutputHeight(), Conv2D_116->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block28/sub3/sub3_conv/Conv2D
        Conv2D_118 = new Convolution("Conv2D_118", 1024,
                        Conv2D_117->getOutputChannels(), Conv2D_117->getOutputHeight(), Conv2D_117->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block28/shortcut/add
        Add_38 = new Sum("Add_38", 
                        Conv2D_118->getOutputChannels(), Conv2D_118->getOutputHeight(), Conv2D_118->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block29/residual_bn_relu/FusedBatchNorm
        BatchNorm_39 = new BatchNorm("BatchNorm_39", 
                        Add_38->getOutputChannels(), Add_38->getOutputHeight(), Add_38->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block29/sub1/sub1_conv/Conv2D
        Conv2D_119 = new Convolution("Conv2D_119", 256,
                        BatchNorm_39->getOutputChannels(), BatchNorm_39->getOutputHeight(), BatchNorm_39->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block29/sub2/sub2_conv/Conv2D
        Conv2D_120 = new Convolution("Conv2D_120", 256,
                        Conv2D_119->getOutputChannels(), Conv2D_119->getOutputHeight(), Conv2D_119->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block29/sub3/sub3_conv/Conv2D
        Conv2D_121 = new Convolution("Conv2D_121", 1024,
                        Conv2D_120->getOutputChannels(), Conv2D_120->getOutputHeight(), Conv2D_120->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block29/shortcut/add
        Add_39 = new Sum("Add_39", 
                        Conv2D_121->getOutputChannels(), Conv2D_121->getOutputHeight(), Conv2D_121->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block30/residual_bn_relu/FusedBatchNorm
        BatchNorm_40 = new BatchNorm("BatchNorm_40", 
                        Add_39->getOutputChannels(), Add_39->getOutputHeight(), Add_39->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block30/sub1/sub1_conv/Conv2D
        Conv2D_122 = new Convolution("Conv2D_122", 256,
                        BatchNorm_40->getOutputChannels(), BatchNorm_40->getOutputHeight(), BatchNorm_40->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block30/sub2/sub2_conv/Conv2D
        Conv2D_123 = new Convolution("Conv2D_123", 256,
                        Conv2D_122->getOutputChannels(), Conv2D_122->getOutputHeight(), Conv2D_122->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block30/sub3/sub3_conv/Conv2D
        Conv2D_124 = new Convolution("Conv2D_124", 1024,
                        Conv2D_123->getOutputChannels(), Conv2D_123->getOutputHeight(), Conv2D_123->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block30/shortcut/add
        Add_40 = new Sum("Add_40", 
                        Conv2D_124->getOutputChannels(), Conv2D_124->getOutputHeight(), Conv2D_124->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer4/block31/residual_bn_relu/FusedBatchNorm
        BatchNorm_41 = new BatchNorm("BatchNorm_41", 
                        Add_40->getOutputChannels(), Add_40->getOutputHeight(), Add_40->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer4/block31/sub1/sub1_conv/Conv2D
        Conv2D_125 = new Convolution("Conv2D_125", 256,
                        BatchNorm_41->getOutputChannels(), BatchNorm_41->getOutputHeight(), BatchNorm_41->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block31/sub2/sub2_conv/Conv2D
        Conv2D_126 = new Convolution("Conv2D_126", 256,
                        Conv2D_125->getOutputChannels(), Conv2D_125->getOutputHeight(), Conv2D_125->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block31/sub3/sub3_conv/Conv2D
        Conv2D_127 = new Convolution("Conv2D_127", 1024,
                        Conv2D_126->getOutputChannels(), Conv2D_126->getOutputHeight(), Conv2D_126->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer4/block31/shortcut/add
        Add_41 = new Sum("Add_41", 
                        Conv2D_127->getOutputChannels(), Conv2D_127->getOutputHeight(), Conv2D_127->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer5/block0/common_bn_relu/FusedBatchNorm
        BatchNorm_42 = new BatchNorm("BatchNorm_42", 
                        Add_41->getOutputChannels(), Add_41->getOutputHeight(), Add_41->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer5/block0/sub1/sub1_conv/Conv2D
        Conv2D_128 = new Convolution("Conv2D_128", 512,
                        BatchNorm_42->getOutputChannels(), BatchNorm_42->getOutputHeight(), BatchNorm_42->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block0/sub2/sub2_conv/Conv2D
        Conv2D_129 = new Convolution("Conv2D_129", 512,
                        Conv2D_128->getOutputChannels(), Conv2D_128->getOutputHeight(), Conv2D_128->getOutputWidth(),
                        3, 3, 0, 1, 0, 1, 2, 2,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block0/sub3/sub3_conv/Conv2D
        Conv2D_130 = new Convolution("Conv2D_130", 2048,
                        Conv2D_129->getOutputChannels(), Conv2D_129->getOutputHeight(), Conv2D_129->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block0/shortcut/sub_sc/Conv2D
        Conv2D_131 = new Convolution("Conv2D_131", 2048,
                        BatchNorm_42->getOutputChannels(), BatchNorm_42->getOutputHeight(), BatchNorm_42->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 2, 2,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block0/shortcut/add
        Add_42 = new Sum("Add_42", 
                        Conv2D_130->getOutputChannels(), Conv2D_130->getOutputHeight(), Conv2D_130->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: layer5/block1/residual_bn_relu/FusedBatchNorm
        BatchNorm_43 = new BatchNorm("BatchNorm_43", 
                        Add_42->getOutputChannels(), Add_42->getOutputHeight(), Add_42->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: layer5/block1/sub1/sub1_conv/Conv2D
        Conv2D_132 = new Convolution("Conv2D_132", 512,
                        BatchNorm_43->getOutputChannels(), BatchNorm_43->getOutputHeight(), BatchNorm_43->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block1/sub2/sub2_conv/Conv2D
        Conv2D_133 = new Convolution("Conv2D_133", 512,
                        Conv2D_132->getOutputChannels(), Conv2D_132->getOutputHeight(), Conv2D_132->getOutputWidth(),
                        3, 3, 1, 1, 1, 1, 1, 1,
                        "Conv2D",
                        "Relu", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block1/sub3/sub3_conv/Conv2D
        Conv2D_134 = new Convolution("Conv2D_134", 2048,
                        Conv2D_133->getOutputChannels(), Conv2D_133->getOutputHeight(), Conv2D_133->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);

        // origin_layer: layer5/block1/shortcut/add
        Add_43 = new Sum("Add_43", 
                        Conv2D_134->getOutputChannels(), Conv2D_134->getOutputHeight(), Conv2D_134->getOutputWidth(),
                        "none",
                        "fp32", "fp32", 1.0, 1.0);

        // origin_layer: avg_fc/fc_bn/FusedBatchNorm
        BatchNorm_44 = new BatchNorm("BatchNorm_44", 
                        Add_43->getOutputChannels(), Add_43->getOutputHeight(), Add_43->getOutputWidth(), 0.0010000000475, "Relu");

        // origin_layer: avg_fc/AvgPool
        AvgPool_1 = new Pooling("AvgPool_1",
                        BatchNorm_44->getOutputChannels(), BatchNorm_44->getOutputHeight(), BatchNorm_44->getOutputWidth(),
                        7, 7, 0, 0, 0, 0, 1, 1,
                        "Avg",
                        BatchNorm_44->getQuantizeType());

        // origin_layer: avg_fc/fc6/Conv2D
        Conv2D_135 = new Convolution("Conv2D_135", 6,
                        AvgPool_1->getOutputChannels(), AvgPool_1->getOutputHeight(), AvgPool_1->getOutputWidth(),
                        1, 1, 0, 0, 0, 0, 1, 1,
                        "Conv2D",
                        "none", 0, 0,
                        "fp32", 0.0, 0.0, 0.0);
    }


    void init_net() {
        Conv2D_1_out = Conv2D_1->Init(batch_size, &cpu_engine, *src_memory, net, Conv2D_1_w, Conv2D_1_b);
        MaxPool_1_out = MaxPool_1->Init(batch_size, &cpu_engine, *Conv2D_1_out, net);
        BatchNorm_1_out = BatchNorm_1->Init(batch_size, &cpu_engine, *MaxPool_1_out, net, BatchNorm_1_weights, BatchNorm_1_mean, BatchNorm_1_variance, MaxPool_1->getFormat());
        Conv2D_2_out = Conv2D_2->Init(batch_size, &cpu_engine, *BatchNorm_1_out, net, Conv2D_2_w, Conv2D_2_b);
        Conv2D_3_out = Conv2D_3->Init(batch_size, &cpu_engine, *Conv2D_2_out, net, Conv2D_3_w, Conv2D_3_b);
        Conv2D_4_out = Conv2D_4->Init(batch_size, &cpu_engine, *Conv2D_3_out, net, Conv2D_4_w, Conv2D_4_b);
        Conv2D_5_out = Conv2D_5->Init(batch_size, &cpu_engine, *BatchNorm_1_out, net, Conv2D_5_w, Conv2D_5_b);
        Add_1_out = Add_1->Init(batch_size, &cpu_engine, Conv2D_4->getFormat(), Conv2D_5->getFormat(), *Conv2D_4_out, *Conv2D_5_out, net);
        BatchNorm_2_out = BatchNorm_2->Init(batch_size, &cpu_engine, *Add_1_out, net, BatchNorm_2_weights, BatchNorm_2_mean, BatchNorm_2_variance, Add_1->getFormat());
        Conv2D_6_out = Conv2D_6->Init(batch_size, &cpu_engine, *BatchNorm_2_out, net, Conv2D_6_w, Conv2D_6_b);
        Conv2D_7_out = Conv2D_7->Init(batch_size, &cpu_engine, *Conv2D_6_out, net, Conv2D_7_w, Conv2D_7_b);
        Conv2D_8_out = Conv2D_8->Init(batch_size, &cpu_engine, *Conv2D_7_out, net, Conv2D_8_w, Conv2D_8_b);
        Add_2_out = Add_2->Init(batch_size, &cpu_engine, Conv2D_8->getFormat(), Add_1->getFormat(), *Conv2D_8_out, *Add_1_out, net);
        BatchNorm_3_out = BatchNorm_3->Init(batch_size, &cpu_engine, *Add_2_out, net, BatchNorm_3_weights, BatchNorm_3_mean, BatchNorm_3_variance, Add_2->getFormat());
        Conv2D_9_out = Conv2D_9->Init(batch_size, &cpu_engine, *BatchNorm_3_out, net, Conv2D_9_w, Conv2D_9_b);
        Conv2D_10_out = Conv2D_10->Init(batch_size, &cpu_engine, *Conv2D_9_out, net, Conv2D_10_w, Conv2D_10_b);
        Conv2D_11_out = Conv2D_11->Init(batch_size, &cpu_engine, *Conv2D_10_out, net, Conv2D_11_w, Conv2D_11_b);
        Conv2D_12_out = Conv2D_12->Init(batch_size, &cpu_engine, *BatchNorm_3_out, net, Conv2D_12_w, Conv2D_12_b);
        Add_3_out = Add_3->Init(batch_size, &cpu_engine, Conv2D_11->getFormat(), Conv2D_12->getFormat(), *Conv2D_11_out, *Conv2D_12_out, net);
        BatchNorm_4_out = BatchNorm_4->Init(batch_size, &cpu_engine, *Add_3_out, net, BatchNorm_4_weights, BatchNorm_4_mean, BatchNorm_4_variance, Add_3->getFormat());
        Conv2D_13_out = Conv2D_13->Init(batch_size, &cpu_engine, *BatchNorm_4_out, net, Conv2D_13_w, Conv2D_13_b);
        Conv2D_14_out = Conv2D_14->Init(batch_size, &cpu_engine, *Conv2D_13_out, net, Conv2D_14_w, Conv2D_14_b);
        Conv2D_15_out = Conv2D_15->Init(batch_size, &cpu_engine, *Conv2D_14_out, net, Conv2D_15_w, Conv2D_15_b);
        Add_4_out = Add_4->Init(batch_size, &cpu_engine, Conv2D_15->getFormat(), Add_3->getFormat(), *Conv2D_15_out, *Add_3_out, net);
        BatchNorm_5_out = BatchNorm_5->Init(batch_size, &cpu_engine, *Add_4_out, net, BatchNorm_5_weights, BatchNorm_5_mean, BatchNorm_5_variance, Add_4->getFormat());
        Conv2D_16_out = Conv2D_16->Init(batch_size, &cpu_engine, *BatchNorm_5_out, net, Conv2D_16_w, Conv2D_16_b);
        Conv2D_17_out = Conv2D_17->Init(batch_size, &cpu_engine, *Conv2D_16_out, net, Conv2D_17_w, Conv2D_17_b);
        Conv2D_18_out = Conv2D_18->Init(batch_size, &cpu_engine, *Conv2D_17_out, net, Conv2D_18_w, Conv2D_18_b);
        Add_5_out = Add_5->Init(batch_size, &cpu_engine, Conv2D_18->getFormat(), Add_4->getFormat(), *Conv2D_18_out, *Add_4_out, net);
        BatchNorm_6_out = BatchNorm_6->Init(batch_size, &cpu_engine, *Add_5_out, net, BatchNorm_6_weights, BatchNorm_6_mean, BatchNorm_6_variance, Add_5->getFormat());
        Conv2D_19_out = Conv2D_19->Init(batch_size, &cpu_engine, *BatchNorm_6_out, net, Conv2D_19_w, Conv2D_19_b);
        Conv2D_20_out = Conv2D_20->Init(batch_size, &cpu_engine, *Conv2D_19_out, net, Conv2D_20_w, Conv2D_20_b);
        Conv2D_21_out = Conv2D_21->Init(batch_size, &cpu_engine, *Conv2D_20_out, net, Conv2D_21_w, Conv2D_21_b);
        Add_6_out = Add_6->Init(batch_size, &cpu_engine, Conv2D_21->getFormat(), Add_5->getFormat(), *Conv2D_21_out, *Add_5_out, net);
        BatchNorm_7_out = BatchNorm_7->Init(batch_size, &cpu_engine, *Add_6_out, net, BatchNorm_7_weights, BatchNorm_7_mean, BatchNorm_7_variance, Add_6->getFormat());
        Conv2D_22_out = Conv2D_22->Init(batch_size, &cpu_engine, *BatchNorm_7_out, net, Conv2D_22_w, Conv2D_22_b);
        Conv2D_23_out = Conv2D_23->Init(batch_size, &cpu_engine, *Conv2D_22_out, net, Conv2D_23_w, Conv2D_23_b);
        Conv2D_24_out = Conv2D_24->Init(batch_size, &cpu_engine, *Conv2D_23_out, net, Conv2D_24_w, Conv2D_24_b);
        Add_7_out = Add_7->Init(batch_size, &cpu_engine, Conv2D_24->getFormat(), Add_6->getFormat(), *Conv2D_24_out, *Add_6_out, net);
        BatchNorm_8_out = BatchNorm_8->Init(batch_size, &cpu_engine, *Add_7_out, net, BatchNorm_8_weights, BatchNorm_8_mean, BatchNorm_8_variance, Add_7->getFormat());
        Conv2D_25_out = Conv2D_25->Init(batch_size, &cpu_engine, *BatchNorm_8_out, net, Conv2D_25_w, Conv2D_25_b);
        Conv2D_26_out = Conv2D_26->Init(batch_size, &cpu_engine, *Conv2D_25_out, net, Conv2D_26_w, Conv2D_26_b);
        Conv2D_27_out = Conv2D_27->Init(batch_size, &cpu_engine, *Conv2D_26_out, net, Conv2D_27_w, Conv2D_27_b);
        Add_8_out = Add_8->Init(batch_size, &cpu_engine, Conv2D_27->getFormat(), Add_7->getFormat(), *Conv2D_27_out, *Add_7_out, net);
        BatchNorm_9_out = BatchNorm_9->Init(batch_size, &cpu_engine, *Add_8_out, net, BatchNorm_9_weights, BatchNorm_9_mean, BatchNorm_9_variance, Add_8->getFormat());
        Conv2D_28_out = Conv2D_28->Init(batch_size, &cpu_engine, *BatchNorm_9_out, net, Conv2D_28_w, Conv2D_28_b);
        Conv2D_29_out = Conv2D_29->Init(batch_size, &cpu_engine, *Conv2D_28_out, net, Conv2D_29_w, Conv2D_29_b);
        Conv2D_30_out = Conv2D_30->Init(batch_size, &cpu_engine, *Conv2D_29_out, net, Conv2D_30_w, Conv2D_30_b);
        Add_9_out = Add_9->Init(batch_size, &cpu_engine, Conv2D_30->getFormat(), Add_8->getFormat(), *Conv2D_30_out, *Add_8_out, net);
        BatchNorm_10_out = BatchNorm_10->Init(batch_size, &cpu_engine, *Add_9_out, net, BatchNorm_10_weights, BatchNorm_10_mean, BatchNorm_10_variance, Add_9->getFormat());
        Conv2D_31_out = Conv2D_31->Init(batch_size, &cpu_engine, *BatchNorm_10_out, net, Conv2D_31_w, Conv2D_31_b);
        Conv2D_32_out = Conv2D_32->Init(batch_size, &cpu_engine, *Conv2D_31_out, net, Conv2D_32_w, Conv2D_32_b);
        Conv2D_33_out = Conv2D_33->Init(batch_size, &cpu_engine, *Conv2D_32_out, net, Conv2D_33_w, Conv2D_33_b);
        Conv2D_34_out = Conv2D_34->Init(batch_size, &cpu_engine, *BatchNorm_10_out, net, Conv2D_34_w, Conv2D_34_b);
        Add_10_out = Add_10->Init(batch_size, &cpu_engine, Conv2D_33->getFormat(), Conv2D_34->getFormat(), *Conv2D_33_out, *Conv2D_34_out, net);
        BatchNorm_11_out = BatchNorm_11->Init(batch_size, &cpu_engine, *Add_10_out, net, BatchNorm_11_weights, BatchNorm_11_mean, BatchNorm_11_variance, Add_10->getFormat());
        Conv2D_35_out = Conv2D_35->Init(batch_size, &cpu_engine, *BatchNorm_11_out, net, Conv2D_35_w, Conv2D_35_b);
        Conv2D_36_out = Conv2D_36->Init(batch_size, &cpu_engine, *Conv2D_35_out, net, Conv2D_36_w, Conv2D_36_b);
        Conv2D_37_out = Conv2D_37->Init(batch_size, &cpu_engine, *Conv2D_36_out, net, Conv2D_37_w, Conv2D_37_b);
        Add_11_out = Add_11->Init(batch_size, &cpu_engine, Conv2D_37->getFormat(), Add_10->getFormat(), *Conv2D_37_out, *Add_10_out, net);
        BatchNorm_12_out = BatchNorm_12->Init(batch_size, &cpu_engine, *Add_11_out, net, BatchNorm_12_weights, BatchNorm_12_mean, BatchNorm_12_variance, Add_11->getFormat());
        Conv2D_38_out = Conv2D_38->Init(batch_size, &cpu_engine, *BatchNorm_12_out, net, Conv2D_38_w, Conv2D_38_b);
        Conv2D_39_out = Conv2D_39->Init(batch_size, &cpu_engine, *Conv2D_38_out, net, Conv2D_39_w, Conv2D_39_b);
        Conv2D_40_out = Conv2D_40->Init(batch_size, &cpu_engine, *Conv2D_39_out, net, Conv2D_40_w, Conv2D_40_b);
        Add_12_out = Add_12->Init(batch_size, &cpu_engine, Conv2D_40->getFormat(), Add_11->getFormat(), *Conv2D_40_out, *Add_11_out, net);
        BatchNorm_13_out = BatchNorm_13->Init(batch_size, &cpu_engine, *Add_12_out, net, BatchNorm_13_weights, BatchNorm_13_mean, BatchNorm_13_variance, Add_12->getFormat());
        Conv2D_41_out = Conv2D_41->Init(batch_size, &cpu_engine, *BatchNorm_13_out, net, Conv2D_41_w, Conv2D_41_b);
        Conv2D_42_out = Conv2D_42->Init(batch_size, &cpu_engine, *Conv2D_41_out, net, Conv2D_42_w, Conv2D_42_b);
        Conv2D_43_out = Conv2D_43->Init(batch_size, &cpu_engine, *Conv2D_42_out, net, Conv2D_43_w, Conv2D_43_b);
        Add_13_out = Add_13->Init(batch_size, &cpu_engine, Conv2D_43->getFormat(), Add_12->getFormat(), *Conv2D_43_out, *Add_12_out, net);
        BatchNorm_14_out = BatchNorm_14->Init(batch_size, &cpu_engine, *Add_13_out, net, BatchNorm_14_weights, BatchNorm_14_mean, BatchNorm_14_variance, Add_13->getFormat());
        Conv2D_44_out = Conv2D_44->Init(batch_size, &cpu_engine, *BatchNorm_14_out, net, Conv2D_44_w, Conv2D_44_b);
        Conv2D_45_out = Conv2D_45->Init(batch_size, &cpu_engine, *Conv2D_44_out, net, Conv2D_45_w, Conv2D_45_b);
        Conv2D_46_out = Conv2D_46->Init(batch_size, &cpu_engine, *Conv2D_45_out, net, Conv2D_46_w, Conv2D_46_b);
        Add_14_out = Add_14->Init(batch_size, &cpu_engine, Conv2D_46->getFormat(), Add_13->getFormat(), *Conv2D_46_out, *Add_13_out, net);
        BatchNorm_15_out = BatchNorm_15->Init(batch_size, &cpu_engine, *Add_14_out, net, BatchNorm_15_weights, BatchNorm_15_mean, BatchNorm_15_variance, Add_14->getFormat());
        Conv2D_47_out = Conv2D_47->Init(batch_size, &cpu_engine, *BatchNorm_15_out, net, Conv2D_47_w, Conv2D_47_b);
        Conv2D_48_out = Conv2D_48->Init(batch_size, &cpu_engine, *Conv2D_47_out, net, Conv2D_48_w, Conv2D_48_b);
        Conv2D_49_out = Conv2D_49->Init(batch_size, &cpu_engine, *Conv2D_48_out, net, Conv2D_49_w, Conv2D_49_b);
        Add_15_out = Add_15->Init(batch_size, &cpu_engine, Conv2D_49->getFormat(), Add_14->getFormat(), *Conv2D_49_out, *Add_14_out, net);
        BatchNorm_16_out = BatchNorm_16->Init(batch_size, &cpu_engine, *Add_15_out, net, BatchNorm_16_weights, BatchNorm_16_mean, BatchNorm_16_variance, Add_15->getFormat());
        Conv2D_50_out = Conv2D_50->Init(batch_size, &cpu_engine, *BatchNorm_16_out, net, Conv2D_50_w, Conv2D_50_b);
        Conv2D_51_out = Conv2D_51->Init(batch_size, &cpu_engine, *Conv2D_50_out, net, Conv2D_51_w, Conv2D_51_b);
        Conv2D_52_out = Conv2D_52->Init(batch_size, &cpu_engine, *Conv2D_51_out, net, Conv2D_52_w, Conv2D_52_b);
        Add_16_out = Add_16->Init(batch_size, &cpu_engine, Conv2D_52->getFormat(), Add_15->getFormat(), *Conv2D_52_out, *Add_15_out, net);
        BatchNorm_17_out = BatchNorm_17->Init(batch_size, &cpu_engine, *Add_16_out, net, BatchNorm_17_weights, BatchNorm_17_mean, BatchNorm_17_variance, Add_16->getFormat());
        Conv2D_53_out = Conv2D_53->Init(batch_size, &cpu_engine, *BatchNorm_17_out, net, Conv2D_53_w, Conv2D_53_b);
        Conv2D_54_out = Conv2D_54->Init(batch_size, &cpu_engine, *Conv2D_53_out, net, Conv2D_54_w, Conv2D_54_b);
        Conv2D_55_out = Conv2D_55->Init(batch_size, &cpu_engine, *Conv2D_54_out, net, Conv2D_55_w, Conv2D_55_b);
        Add_17_out = Add_17->Init(batch_size, &cpu_engine, Conv2D_55->getFormat(), Add_16->getFormat(), *Conv2D_55_out, *Add_16_out, net);
        BatchNorm_18_out = BatchNorm_18->Init(batch_size, &cpu_engine, *Add_17_out, net, BatchNorm_18_weights, BatchNorm_18_mean, BatchNorm_18_variance, Add_17->getFormat());
        Conv2D_56_out = Conv2D_56->Init(batch_size, &cpu_engine, *BatchNorm_18_out, net, Conv2D_56_w, Conv2D_56_b);
        Conv2D_57_out = Conv2D_57->Init(batch_size, &cpu_engine, *Conv2D_56_out, net, Conv2D_57_w, Conv2D_57_b);
        Conv2D_58_out = Conv2D_58->Init(batch_size, &cpu_engine, *Conv2D_57_out, net, Conv2D_58_w, Conv2D_58_b);
        Add_18_out = Add_18->Init(batch_size, &cpu_engine, Conv2D_58->getFormat(), Add_17->getFormat(), *Conv2D_58_out, *Add_17_out, net);
        BatchNorm_19_out = BatchNorm_19->Init(batch_size, &cpu_engine, *Add_18_out, net, BatchNorm_19_weights, BatchNorm_19_mean, BatchNorm_19_variance, Add_18->getFormat());
        Conv2D_59_out = Conv2D_59->Init(batch_size, &cpu_engine, *BatchNorm_19_out, net, Conv2D_59_w, Conv2D_59_b);
        Conv2D_60_out = Conv2D_60->Init(batch_size, &cpu_engine, *Conv2D_59_out, net, Conv2D_60_w, Conv2D_60_b);
        Conv2D_61_out = Conv2D_61->Init(batch_size, &cpu_engine, *Conv2D_60_out, net, Conv2D_61_w, Conv2D_61_b);
        Add_19_out = Add_19->Init(batch_size, &cpu_engine, Conv2D_61->getFormat(), Add_18->getFormat(), *Conv2D_61_out, *Add_18_out, net);
        BatchNorm_20_out = BatchNorm_20->Init(batch_size, &cpu_engine, *Add_19_out, net, BatchNorm_20_weights, BatchNorm_20_mean, BatchNorm_20_variance, Add_19->getFormat());
        Conv2D_62_out = Conv2D_62->Init(batch_size, &cpu_engine, *BatchNorm_20_out, net, Conv2D_62_w, Conv2D_62_b);
        Conv2D_63_out = Conv2D_63->Init(batch_size, &cpu_engine, *Conv2D_62_out, net, Conv2D_63_w, Conv2D_63_b);
        Conv2D_64_out = Conv2D_64->Init(batch_size, &cpu_engine, *Conv2D_63_out, net, Conv2D_64_w, Conv2D_64_b);
        Add_20_out = Add_20->Init(batch_size, &cpu_engine, Conv2D_64->getFormat(), Add_19->getFormat(), *Conv2D_64_out, *Add_19_out, net);
        BatchNorm_21_out = BatchNorm_21->Init(batch_size, &cpu_engine, *Add_20_out, net, BatchNorm_21_weights, BatchNorm_21_mean, BatchNorm_21_variance, Add_20->getFormat());
        Conv2D_65_out = Conv2D_65->Init(batch_size, &cpu_engine, *BatchNorm_21_out, net, Conv2D_65_w, Conv2D_65_b);
        Conv2D_66_out = Conv2D_66->Init(batch_size, &cpu_engine, *Conv2D_65_out, net, Conv2D_66_w, Conv2D_66_b);
        Conv2D_67_out = Conv2D_67->Init(batch_size, &cpu_engine, *Conv2D_66_out, net, Conv2D_67_w, Conv2D_67_b);
        Add_21_out = Add_21->Init(batch_size, &cpu_engine, Conv2D_67->getFormat(), Add_20->getFormat(), *Conv2D_67_out, *Add_20_out, net);
        BatchNorm_22_out = BatchNorm_22->Init(batch_size, &cpu_engine, *Add_21_out, net, BatchNorm_22_weights, BatchNorm_22_mean, BatchNorm_22_variance, Add_21->getFormat());
        Conv2D_68_out = Conv2D_68->Init(batch_size, &cpu_engine, *BatchNorm_22_out, net, Conv2D_68_w, Conv2D_68_b);
        Conv2D_69_out = Conv2D_69->Init(batch_size, &cpu_engine, *Conv2D_68_out, net, Conv2D_69_w, Conv2D_69_b);
        Conv2D_70_out = Conv2D_70->Init(batch_size, &cpu_engine, *Conv2D_69_out, net, Conv2D_70_w, Conv2D_70_b);
        Add_22_out = Add_22->Init(batch_size, &cpu_engine, Conv2D_70->getFormat(), Add_21->getFormat(), *Conv2D_70_out, *Add_21_out, net);
        BatchNorm_23_out = BatchNorm_23->Init(batch_size, &cpu_engine, *Add_22_out, net, BatchNorm_23_weights, BatchNorm_23_mean, BatchNorm_23_variance, Add_22->getFormat());
        Conv2D_71_out = Conv2D_71->Init(batch_size, &cpu_engine, *BatchNorm_23_out, net, Conv2D_71_w, Conv2D_71_b);
        Conv2D_72_out = Conv2D_72->Init(batch_size, &cpu_engine, *Conv2D_71_out, net, Conv2D_72_w, Conv2D_72_b);
        Conv2D_73_out = Conv2D_73->Init(batch_size, &cpu_engine, *Conv2D_72_out, net, Conv2D_73_w, Conv2D_73_b);
        Add_23_out = Add_23->Init(batch_size, &cpu_engine, Conv2D_73->getFormat(), Add_22->getFormat(), *Conv2D_73_out, *Add_22_out, net);
        BatchNorm_24_out = BatchNorm_24->Init(batch_size, &cpu_engine, *Add_23_out, net, BatchNorm_24_weights, BatchNorm_24_mean, BatchNorm_24_variance, Add_23->getFormat());
        Conv2D_74_out = Conv2D_74->Init(batch_size, &cpu_engine, *BatchNorm_24_out, net, Conv2D_74_w, Conv2D_74_b);
        Conv2D_75_out = Conv2D_75->Init(batch_size, &cpu_engine, *Conv2D_74_out, net, Conv2D_75_w, Conv2D_75_b);
        Conv2D_76_out = Conv2D_76->Init(batch_size, &cpu_engine, *Conv2D_75_out, net, Conv2D_76_w, Conv2D_76_b);
        Add_24_out = Add_24->Init(batch_size, &cpu_engine, Conv2D_76->getFormat(), Add_23->getFormat(), *Conv2D_76_out, *Add_23_out, net);
        BatchNorm_25_out = BatchNorm_25->Init(batch_size, &cpu_engine, *Add_24_out, net, BatchNorm_25_weights, BatchNorm_25_mean, BatchNorm_25_variance, Add_24->getFormat());
        Conv2D_77_out = Conv2D_77->Init(batch_size, &cpu_engine, *BatchNorm_25_out, net, Conv2D_77_w, Conv2D_77_b);
        Conv2D_78_out = Conv2D_78->Init(batch_size, &cpu_engine, *Conv2D_77_out, net, Conv2D_78_w, Conv2D_78_b);
        Conv2D_79_out = Conv2D_79->Init(batch_size, &cpu_engine, *Conv2D_78_out, net, Conv2D_79_w, Conv2D_79_b);
        Add_25_out = Add_25->Init(batch_size, &cpu_engine, Conv2D_79->getFormat(), Add_24->getFormat(), *Conv2D_79_out, *Add_24_out, net);
        BatchNorm_26_out = BatchNorm_26->Init(batch_size, &cpu_engine, *Add_25_out, net, BatchNorm_26_weights, BatchNorm_26_mean, BatchNorm_26_variance, Add_25->getFormat());
        Conv2D_80_out = Conv2D_80->Init(batch_size, &cpu_engine, *BatchNorm_26_out, net, Conv2D_80_w, Conv2D_80_b);
        Conv2D_81_out = Conv2D_81->Init(batch_size, &cpu_engine, *Conv2D_80_out, net, Conv2D_81_w, Conv2D_81_b);
        Conv2D_82_out = Conv2D_82->Init(batch_size, &cpu_engine, *Conv2D_81_out, net, Conv2D_82_w, Conv2D_82_b);
        Add_26_out = Add_26->Init(batch_size, &cpu_engine, Conv2D_82->getFormat(), Add_25->getFormat(), *Conv2D_82_out, *Add_25_out, net);
        BatchNorm_27_out = BatchNorm_27->Init(batch_size, &cpu_engine, *Add_26_out, net, BatchNorm_27_weights, BatchNorm_27_mean, BatchNorm_27_variance, Add_26->getFormat());
        Conv2D_83_out = Conv2D_83->Init(batch_size, &cpu_engine, *BatchNorm_27_out, net, Conv2D_83_w, Conv2D_83_b);
        Conv2D_84_out = Conv2D_84->Init(batch_size, &cpu_engine, *Conv2D_83_out, net, Conv2D_84_w, Conv2D_84_b);
        Conv2D_85_out = Conv2D_85->Init(batch_size, &cpu_engine, *Conv2D_84_out, net, Conv2D_85_w, Conv2D_85_b);
        Add_27_out = Add_27->Init(batch_size, &cpu_engine, Conv2D_85->getFormat(), Add_26->getFormat(), *Conv2D_85_out, *Add_26_out, net);
        BatchNorm_28_out = BatchNorm_28->Init(batch_size, &cpu_engine, *Add_27_out, net, BatchNorm_28_weights, BatchNorm_28_mean, BatchNorm_28_variance, Add_27->getFormat());
        Conv2D_86_out = Conv2D_86->Init(batch_size, &cpu_engine, *BatchNorm_28_out, net, Conv2D_86_w, Conv2D_86_b);
        Conv2D_87_out = Conv2D_87->Init(batch_size, &cpu_engine, *Conv2D_86_out, net, Conv2D_87_w, Conv2D_87_b);
        Conv2D_88_out = Conv2D_88->Init(batch_size, &cpu_engine, *Conv2D_87_out, net, Conv2D_88_w, Conv2D_88_b);
        Add_28_out = Add_28->Init(batch_size, &cpu_engine, Conv2D_88->getFormat(), Add_27->getFormat(), *Conv2D_88_out, *Add_27_out, net);
        BatchNorm_29_out = BatchNorm_29->Init(batch_size, &cpu_engine, *Add_28_out, net, BatchNorm_29_weights, BatchNorm_29_mean, BatchNorm_29_variance, Add_28->getFormat());
        Conv2D_89_out = Conv2D_89->Init(batch_size, &cpu_engine, *BatchNorm_29_out, net, Conv2D_89_w, Conv2D_89_b);
        Conv2D_90_out = Conv2D_90->Init(batch_size, &cpu_engine, *Conv2D_89_out, net, Conv2D_90_w, Conv2D_90_b);
        Conv2D_91_out = Conv2D_91->Init(batch_size, &cpu_engine, *Conv2D_90_out, net, Conv2D_91_w, Conv2D_91_b);
        Add_29_out = Add_29->Init(batch_size, &cpu_engine, Conv2D_91->getFormat(), Add_28->getFormat(), *Conv2D_91_out, *Add_28_out, net);
        BatchNorm_30_out = BatchNorm_30->Init(batch_size, &cpu_engine, *Add_29_out, net, BatchNorm_30_weights, BatchNorm_30_mean, BatchNorm_30_variance, Add_29->getFormat());
        Conv2D_92_out = Conv2D_92->Init(batch_size, &cpu_engine, *BatchNorm_30_out, net, Conv2D_92_w, Conv2D_92_b);
        Conv2D_93_out = Conv2D_93->Init(batch_size, &cpu_engine, *Conv2D_92_out, net, Conv2D_93_w, Conv2D_93_b);
        Conv2D_94_out = Conv2D_94->Init(batch_size, &cpu_engine, *Conv2D_93_out, net, Conv2D_94_w, Conv2D_94_b);
        Add_30_out = Add_30->Init(batch_size, &cpu_engine, Conv2D_94->getFormat(), Add_29->getFormat(), *Conv2D_94_out, *Add_29_out, net);
        BatchNorm_31_out = BatchNorm_31->Init(batch_size, &cpu_engine, *Add_30_out, net, BatchNorm_31_weights, BatchNorm_31_mean, BatchNorm_31_variance, Add_30->getFormat());
        Conv2D_95_out = Conv2D_95->Init(batch_size, &cpu_engine, *BatchNorm_31_out, net, Conv2D_95_w, Conv2D_95_b);
        Conv2D_96_out = Conv2D_96->Init(batch_size, &cpu_engine, *Conv2D_95_out, net, Conv2D_96_w, Conv2D_96_b);
        Conv2D_97_out = Conv2D_97->Init(batch_size, &cpu_engine, *Conv2D_96_out, net, Conv2D_97_w, Conv2D_97_b);
        Add_31_out = Add_31->Init(batch_size, &cpu_engine, Conv2D_97->getFormat(), Add_30->getFormat(), *Conv2D_97_out, *Add_30_out, net);
        BatchNorm_32_out = BatchNorm_32->Init(batch_size, &cpu_engine, *Add_31_out, net, BatchNorm_32_weights, BatchNorm_32_mean, BatchNorm_32_variance, Add_31->getFormat());
        Conv2D_98_out = Conv2D_98->Init(batch_size, &cpu_engine, *BatchNorm_32_out, net, Conv2D_98_w, Conv2D_98_b);
        Conv2D_99_out = Conv2D_99->Init(batch_size, &cpu_engine, *Conv2D_98_out, net, Conv2D_99_w, Conv2D_99_b);
        Conv2D_100_out = Conv2D_100->Init(batch_size, &cpu_engine, *Conv2D_99_out, net, Conv2D_100_w, Conv2D_100_b);
        Add_32_out = Add_32->Init(batch_size, &cpu_engine, Conv2D_100->getFormat(), Add_31->getFormat(), *Conv2D_100_out, *Add_31_out, net);
        BatchNorm_33_out = BatchNorm_33->Init(batch_size, &cpu_engine, *Add_32_out, net, BatchNorm_33_weights, BatchNorm_33_mean, BatchNorm_33_variance, Add_32->getFormat());
        Conv2D_101_out = Conv2D_101->Init(batch_size, &cpu_engine, *BatchNorm_33_out, net, Conv2D_101_w, Conv2D_101_b);
        Conv2D_102_out = Conv2D_102->Init(batch_size, &cpu_engine, *Conv2D_101_out, net, Conv2D_102_w, Conv2D_102_b);
        Conv2D_103_out = Conv2D_103->Init(batch_size, &cpu_engine, *Conv2D_102_out, net, Conv2D_103_w, Conv2D_103_b);
        Add_33_out = Add_33->Init(batch_size, &cpu_engine, Conv2D_103->getFormat(), Add_32->getFormat(), *Conv2D_103_out, *Add_32_out, net);
        BatchNorm_34_out = BatchNorm_34->Init(batch_size, &cpu_engine, *Add_33_out, net, BatchNorm_34_weights, BatchNorm_34_mean, BatchNorm_34_variance, Add_33->getFormat());
        Conv2D_104_out = Conv2D_104->Init(batch_size, &cpu_engine, *BatchNorm_34_out, net, Conv2D_104_w, Conv2D_104_b);
        Conv2D_105_out = Conv2D_105->Init(batch_size, &cpu_engine, *Conv2D_104_out, net, Conv2D_105_w, Conv2D_105_b);
        Conv2D_106_out = Conv2D_106->Init(batch_size, &cpu_engine, *Conv2D_105_out, net, Conv2D_106_w, Conv2D_106_b);
        Add_34_out = Add_34->Init(batch_size, &cpu_engine, Conv2D_106->getFormat(), Add_33->getFormat(), *Conv2D_106_out, *Add_33_out, net);
        BatchNorm_35_out = BatchNorm_35->Init(batch_size, &cpu_engine, *Add_34_out, net, BatchNorm_35_weights, BatchNorm_35_mean, BatchNorm_35_variance, Add_34->getFormat());
        Conv2D_107_out = Conv2D_107->Init(batch_size, &cpu_engine, *BatchNorm_35_out, net, Conv2D_107_w, Conv2D_107_b);
        Conv2D_108_out = Conv2D_108->Init(batch_size, &cpu_engine, *Conv2D_107_out, net, Conv2D_108_w, Conv2D_108_b);
        Conv2D_109_out = Conv2D_109->Init(batch_size, &cpu_engine, *Conv2D_108_out, net, Conv2D_109_w, Conv2D_109_b);
        Add_35_out = Add_35->Init(batch_size, &cpu_engine, Conv2D_109->getFormat(), Add_34->getFormat(), *Conv2D_109_out, *Add_34_out, net);
        BatchNorm_36_out = BatchNorm_36->Init(batch_size, &cpu_engine, *Add_35_out, net, BatchNorm_36_weights, BatchNorm_36_mean, BatchNorm_36_variance, Add_35->getFormat());
        Conv2D_110_out = Conv2D_110->Init(batch_size, &cpu_engine, *BatchNorm_36_out, net, Conv2D_110_w, Conv2D_110_b);
        Conv2D_111_out = Conv2D_111->Init(batch_size, &cpu_engine, *Conv2D_110_out, net, Conv2D_111_w, Conv2D_111_b);
        Conv2D_112_out = Conv2D_112->Init(batch_size, &cpu_engine, *Conv2D_111_out, net, Conv2D_112_w, Conv2D_112_b);
        Add_36_out = Add_36->Init(batch_size, &cpu_engine, Conv2D_112->getFormat(), Add_35->getFormat(), *Conv2D_112_out, *Add_35_out, net);
        BatchNorm_37_out = BatchNorm_37->Init(batch_size, &cpu_engine, *Add_36_out, net, BatchNorm_37_weights, BatchNorm_37_mean, BatchNorm_37_variance, Add_36->getFormat());
        Conv2D_113_out = Conv2D_113->Init(batch_size, &cpu_engine, *BatchNorm_37_out, net, Conv2D_113_w, Conv2D_113_b);
        Conv2D_114_out = Conv2D_114->Init(batch_size, &cpu_engine, *Conv2D_113_out, net, Conv2D_114_w, Conv2D_114_b);
        Conv2D_115_out = Conv2D_115->Init(batch_size, &cpu_engine, *Conv2D_114_out, net, Conv2D_115_w, Conv2D_115_b);
        Add_37_out = Add_37->Init(batch_size, &cpu_engine, Conv2D_115->getFormat(), Add_36->getFormat(), *Conv2D_115_out, *Add_36_out, net);
        BatchNorm_38_out = BatchNorm_38->Init(batch_size, &cpu_engine, *Add_37_out, net, BatchNorm_38_weights, BatchNorm_38_mean, BatchNorm_38_variance, Add_37->getFormat());
        Conv2D_116_out = Conv2D_116->Init(batch_size, &cpu_engine, *BatchNorm_38_out, net, Conv2D_116_w, Conv2D_116_b);
        Conv2D_117_out = Conv2D_117->Init(batch_size, &cpu_engine, *Conv2D_116_out, net, Conv2D_117_w, Conv2D_117_b);
        Conv2D_118_out = Conv2D_118->Init(batch_size, &cpu_engine, *Conv2D_117_out, net, Conv2D_118_w, Conv2D_118_b);
        Add_38_out = Add_38->Init(batch_size, &cpu_engine, Conv2D_118->getFormat(), Add_37->getFormat(), *Conv2D_118_out, *Add_37_out, net);
        BatchNorm_39_out = BatchNorm_39->Init(batch_size, &cpu_engine, *Add_38_out, net, BatchNorm_39_weights, BatchNorm_39_mean, BatchNorm_39_variance, Add_38->getFormat());
        Conv2D_119_out = Conv2D_119->Init(batch_size, &cpu_engine, *BatchNorm_39_out, net, Conv2D_119_w, Conv2D_119_b);
        Conv2D_120_out = Conv2D_120->Init(batch_size, &cpu_engine, *Conv2D_119_out, net, Conv2D_120_w, Conv2D_120_b);
        Conv2D_121_out = Conv2D_121->Init(batch_size, &cpu_engine, *Conv2D_120_out, net, Conv2D_121_w, Conv2D_121_b);
        Add_39_out = Add_39->Init(batch_size, &cpu_engine, Conv2D_121->getFormat(), Add_38->getFormat(), *Conv2D_121_out, *Add_38_out, net);
        BatchNorm_40_out = BatchNorm_40->Init(batch_size, &cpu_engine, *Add_39_out, net, BatchNorm_40_weights, BatchNorm_40_mean, BatchNorm_40_variance, Add_39->getFormat());
        Conv2D_122_out = Conv2D_122->Init(batch_size, &cpu_engine, *BatchNorm_40_out, net, Conv2D_122_w, Conv2D_122_b);
        Conv2D_123_out = Conv2D_123->Init(batch_size, &cpu_engine, *Conv2D_122_out, net, Conv2D_123_w, Conv2D_123_b);
        Conv2D_124_out = Conv2D_124->Init(batch_size, &cpu_engine, *Conv2D_123_out, net, Conv2D_124_w, Conv2D_124_b);
        Add_40_out = Add_40->Init(batch_size, &cpu_engine, Conv2D_124->getFormat(), Add_39->getFormat(), *Conv2D_124_out, *Add_39_out, net);
        BatchNorm_41_out = BatchNorm_41->Init(batch_size, &cpu_engine, *Add_40_out, net, BatchNorm_41_weights, BatchNorm_41_mean, BatchNorm_41_variance, Add_40->getFormat());
        Conv2D_125_out = Conv2D_125->Init(batch_size, &cpu_engine, *BatchNorm_41_out, net, Conv2D_125_w, Conv2D_125_b);
        Conv2D_126_out = Conv2D_126->Init(batch_size, &cpu_engine, *Conv2D_125_out, net, Conv2D_126_w, Conv2D_126_b);
        Conv2D_127_out = Conv2D_127->Init(batch_size, &cpu_engine, *Conv2D_126_out, net, Conv2D_127_w, Conv2D_127_b);
        Add_41_out = Add_41->Init(batch_size, &cpu_engine, Conv2D_127->getFormat(), Add_40->getFormat(), *Conv2D_127_out, *Add_40_out, net);
        BatchNorm_42_out = BatchNorm_42->Init(batch_size, &cpu_engine, *Add_41_out, net, BatchNorm_42_weights, BatchNorm_42_mean, BatchNorm_42_variance, Add_41->getFormat());
        Conv2D_128_out = Conv2D_128->Init(batch_size, &cpu_engine, *BatchNorm_42_out, net, Conv2D_128_w, Conv2D_128_b);
        Conv2D_129_out = Conv2D_129->Init(batch_size, &cpu_engine, *Conv2D_128_out, net, Conv2D_129_w, Conv2D_129_b);
        Conv2D_130_out = Conv2D_130->Init(batch_size, &cpu_engine, *Conv2D_129_out, net, Conv2D_130_w, Conv2D_130_b);
        Conv2D_131_out = Conv2D_131->Init(batch_size, &cpu_engine, *BatchNorm_42_out, net, Conv2D_131_w, Conv2D_131_b);
        Add_42_out = Add_42->Init(batch_size, &cpu_engine, Conv2D_130->getFormat(), Conv2D_131->getFormat(), *Conv2D_130_out, *Conv2D_131_out, net);
        BatchNorm_43_out = BatchNorm_43->Init(batch_size, &cpu_engine, *Add_42_out, net, BatchNorm_43_weights, BatchNorm_43_mean, BatchNorm_43_variance, Add_42->getFormat());
        Conv2D_132_out = Conv2D_132->Init(batch_size, &cpu_engine, *BatchNorm_43_out, net, Conv2D_132_w, Conv2D_132_b);
        Conv2D_133_out = Conv2D_133->Init(batch_size, &cpu_engine, *Conv2D_132_out, net, Conv2D_133_w, Conv2D_133_b);
        Conv2D_134_out = Conv2D_134->Init(batch_size, &cpu_engine, *Conv2D_133_out, net, Conv2D_134_w, Conv2D_134_b);
        Add_43_out = Add_43->Init(batch_size, &cpu_engine, Conv2D_134->getFormat(), Add_42->getFormat(), *Conv2D_134_out, *Add_42_out, net);
        BatchNorm_44_out = BatchNorm_44->Init(batch_size, &cpu_engine, *Add_43_out, net, BatchNorm_44_weights, BatchNorm_44_mean, BatchNorm_44_variance, Add_43->getFormat());
        AvgPool_1_out = AvgPool_1->Init(batch_size, &cpu_engine, *BatchNorm_44_out, net);
        Conv2D_135_out = Conv2D_135->Init(batch_size, &cpu_engine, *AvgPool_1_out, net, Conv2D_135_w, Conv2D_135_b);
    }


    void read_weights() {
        FILE *fp;
        fp = fopen(weights_path, "rb");
        if(fp==0){ printf("ERROR: Fail to open [%s]\n", weights_path); exit(0);}

        // origin_layer: layer1/layer1_conv/Conv2D
        fread(Conv2D_1_w, sizeof(float),
              Conv2D_1->getKernelHeight() * Conv2D_1->getKernelWidth() * Conv2D_1->getOutputChannels() * Conv2D_1->getInputChannels(), fp);
        fread(Conv2D_1_b, sizeof(float), Conv2D_1->getOutputChannels(), fp);

        // origin_layer: layer2/block0/common_bn_relu/FusedBatchNorm
        fread(BatchNorm_1_mean, sizeof(float), BatchNorm_1->getOutputChannels(), fp);
        fread(BatchNorm_1_variance, sizeof(float), BatchNorm_1->getOutputChannels(), fp);
        fread(BatchNorm_1_weights, sizeof(float), BatchNorm_1->getOutputChannels() * 2, fp);

        // origin_layer: layer2/block0/sub1/sub1_conv/Conv2D
        fread(Conv2D_2_w, sizeof(float),
              Conv2D_2->getKernelHeight() * Conv2D_2->getKernelWidth() * Conv2D_2->getOutputChannels() * Conv2D_2->getInputChannels(), fp);
        fread(Conv2D_2_b, sizeof(float), Conv2D_2->getOutputChannels(), fp);

        // origin_layer: layer2/block0/sub2/sub2_conv/Conv2D
        fread(Conv2D_3_w, sizeof(float),
              Conv2D_3->getKernelHeight() * Conv2D_3->getKernelWidth() * Conv2D_3->getOutputChannels() * Conv2D_3->getInputChannels(), fp);
        fread(Conv2D_3_b, sizeof(float), Conv2D_3->getOutputChannels(), fp);

        // origin_layer: layer2/block0/sub3/sub3_conv/Conv2D
        fread(Conv2D_4_w, sizeof(float),
              Conv2D_4->getKernelHeight() * Conv2D_4->getKernelWidth() * Conv2D_4->getOutputChannels() * Conv2D_4->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_4->getOutputChannels(); i ++) Conv2D_4_b[i] = 0.0;

        // origin_layer: layer2/block0/shortcut/sub_sc/Conv2D
        fread(Conv2D_5_w, sizeof(float),
              Conv2D_5->getKernelHeight() * Conv2D_5->getKernelWidth() * Conv2D_5->getOutputChannels() * Conv2D_5->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_5->getOutputChannels(); i ++) Conv2D_5_b[i] = 0.0;

        // origin_layer: layer2/block1/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_2_mean, sizeof(float), BatchNorm_2->getOutputChannels(), fp);
        fread(BatchNorm_2_variance, sizeof(float), BatchNorm_2->getOutputChannels(), fp);
        fread(BatchNorm_2_weights, sizeof(float), BatchNorm_2->getOutputChannels() * 2, fp);

        // origin_layer: layer2/block1/sub1/sub1_conv/Conv2D
        fread(Conv2D_6_w, sizeof(float),
              Conv2D_6->getKernelHeight() * Conv2D_6->getKernelWidth() * Conv2D_6->getOutputChannels() * Conv2D_6->getInputChannels(), fp);
        fread(Conv2D_6_b, sizeof(float), Conv2D_6->getOutputChannels(), fp);

        // origin_layer: layer2/block1/sub2/sub2_conv/Conv2D
        fread(Conv2D_7_w, sizeof(float),
              Conv2D_7->getKernelHeight() * Conv2D_7->getKernelWidth() * Conv2D_7->getOutputChannels() * Conv2D_7->getInputChannels(), fp);
        fread(Conv2D_7_b, sizeof(float), Conv2D_7->getOutputChannels(), fp);

        // origin_layer: layer2/block1/sub3/sub3_conv/Conv2D
        fread(Conv2D_8_w, sizeof(float),
              Conv2D_8->getKernelHeight() * Conv2D_8->getKernelWidth() * Conv2D_8->getOutputChannels() * Conv2D_8->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_8->getOutputChannels(); i ++) Conv2D_8_b[i] = 0.0;

        // origin_layer: layer3/block0/common_bn_relu/FusedBatchNorm
        fread(BatchNorm_3_mean, sizeof(float), BatchNorm_3->getOutputChannels(), fp);
        fread(BatchNorm_3_variance, sizeof(float), BatchNorm_3->getOutputChannels(), fp);
        fread(BatchNorm_3_weights, sizeof(float), BatchNorm_3->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block0/sub1/sub1_conv/Conv2D
        fread(Conv2D_9_w, sizeof(float),
              Conv2D_9->getKernelHeight() * Conv2D_9->getKernelWidth() * Conv2D_9->getOutputChannels() * Conv2D_9->getInputChannels(), fp);
        fread(Conv2D_9_b, sizeof(float), Conv2D_9->getOutputChannels(), fp);

        // origin_layer: layer3/block0/sub2/sub2_conv/Conv2D
        fread(Conv2D_10_w, sizeof(float),
              Conv2D_10->getKernelHeight() * Conv2D_10->getKernelWidth() * Conv2D_10->getOutputChannels() * Conv2D_10->getInputChannels(), fp);
        fread(Conv2D_10_b, sizeof(float), Conv2D_10->getOutputChannels(), fp);

        // origin_layer: layer3/block0/sub3/sub3_conv/Conv2D
        fread(Conv2D_11_w, sizeof(float),
              Conv2D_11->getKernelHeight() * Conv2D_11->getKernelWidth() * Conv2D_11->getOutputChannels() * Conv2D_11->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_11->getOutputChannels(); i ++) Conv2D_11_b[i] = 0.0;

        // origin_layer: layer3/block0/shortcut/sub_sc/Conv2D
        fread(Conv2D_12_w, sizeof(float),
              Conv2D_12->getKernelHeight() * Conv2D_12->getKernelWidth() * Conv2D_12->getOutputChannels() * Conv2D_12->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_12->getOutputChannels(); i ++) Conv2D_12_b[i] = 0.0;

        // origin_layer: layer3/block1/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_4_mean, sizeof(float), BatchNorm_4->getOutputChannels(), fp);
        fread(BatchNorm_4_variance, sizeof(float), BatchNorm_4->getOutputChannels(), fp);
        fread(BatchNorm_4_weights, sizeof(float), BatchNorm_4->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block1/sub1/sub1_conv/Conv2D
        fread(Conv2D_13_w, sizeof(float),
              Conv2D_13->getKernelHeight() * Conv2D_13->getKernelWidth() * Conv2D_13->getOutputChannels() * Conv2D_13->getInputChannels(), fp);
        fread(Conv2D_13_b, sizeof(float), Conv2D_13->getOutputChannels(), fp);

        // origin_layer: layer3/block1/sub2/sub2_conv/Conv2D
        fread(Conv2D_14_w, sizeof(float),
              Conv2D_14->getKernelHeight() * Conv2D_14->getKernelWidth() * Conv2D_14->getOutputChannels() * Conv2D_14->getInputChannels(), fp);
        fread(Conv2D_14_b, sizeof(float), Conv2D_14->getOutputChannels(), fp);

        // origin_layer: layer3/block1/sub3/sub3_conv/Conv2D
        fread(Conv2D_15_w, sizeof(float),
              Conv2D_15->getKernelHeight() * Conv2D_15->getKernelWidth() * Conv2D_15->getOutputChannels() * Conv2D_15->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_15->getOutputChannels(); i ++) Conv2D_15_b[i] = 0.0;

        // origin_layer: layer3/block2/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_5_mean, sizeof(float), BatchNorm_5->getOutputChannels(), fp);
        fread(BatchNorm_5_variance, sizeof(float), BatchNorm_5->getOutputChannels(), fp);
        fread(BatchNorm_5_weights, sizeof(float), BatchNorm_5->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block2/sub1/sub1_conv/Conv2D
        fread(Conv2D_16_w, sizeof(float),
              Conv2D_16->getKernelHeight() * Conv2D_16->getKernelWidth() * Conv2D_16->getOutputChannels() * Conv2D_16->getInputChannels(), fp);
        fread(Conv2D_16_b, sizeof(float), Conv2D_16->getOutputChannels(), fp);

        // origin_layer: layer3/block2/sub2/sub2_conv/Conv2D
        fread(Conv2D_17_w, sizeof(float),
              Conv2D_17->getKernelHeight() * Conv2D_17->getKernelWidth() * Conv2D_17->getOutputChannels() * Conv2D_17->getInputChannels(), fp);
        fread(Conv2D_17_b, sizeof(float), Conv2D_17->getOutputChannels(), fp);

        // origin_layer: layer3/block2/sub3/sub3_conv/Conv2D
        fread(Conv2D_18_w, sizeof(float),
              Conv2D_18->getKernelHeight() * Conv2D_18->getKernelWidth() * Conv2D_18->getOutputChannels() * Conv2D_18->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_18->getOutputChannels(); i ++) Conv2D_18_b[i] = 0.0;

        // origin_layer: layer3/block3/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_6_mean, sizeof(float), BatchNorm_6->getOutputChannels(), fp);
        fread(BatchNorm_6_variance, sizeof(float), BatchNorm_6->getOutputChannels(), fp);
        fread(BatchNorm_6_weights, sizeof(float), BatchNorm_6->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block3/sub1/sub1_conv/Conv2D
        fread(Conv2D_19_w, sizeof(float),
              Conv2D_19->getKernelHeight() * Conv2D_19->getKernelWidth() * Conv2D_19->getOutputChannels() * Conv2D_19->getInputChannels(), fp);
        fread(Conv2D_19_b, sizeof(float), Conv2D_19->getOutputChannels(), fp);

        // origin_layer: layer3/block3/sub2/sub2_conv/Conv2D
        fread(Conv2D_20_w, sizeof(float),
              Conv2D_20->getKernelHeight() * Conv2D_20->getKernelWidth() * Conv2D_20->getOutputChannels() * Conv2D_20->getInputChannels(), fp);
        fread(Conv2D_20_b, sizeof(float), Conv2D_20->getOutputChannels(), fp);

        // origin_layer: layer3/block3/sub3/sub3_conv/Conv2D
        fread(Conv2D_21_w, sizeof(float),
              Conv2D_21->getKernelHeight() * Conv2D_21->getKernelWidth() * Conv2D_21->getOutputChannels() * Conv2D_21->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_21->getOutputChannels(); i ++) Conv2D_21_b[i] = 0.0;

        // origin_layer: layer3/block4/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_7_mean, sizeof(float), BatchNorm_7->getOutputChannels(), fp);
        fread(BatchNorm_7_variance, sizeof(float), BatchNorm_7->getOutputChannels(), fp);
        fread(BatchNorm_7_weights, sizeof(float), BatchNorm_7->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block4/sub1/sub1_conv/Conv2D
        fread(Conv2D_22_w, sizeof(float),
              Conv2D_22->getKernelHeight() * Conv2D_22->getKernelWidth() * Conv2D_22->getOutputChannels() * Conv2D_22->getInputChannels(), fp);
        fread(Conv2D_22_b, sizeof(float), Conv2D_22->getOutputChannels(), fp);

        // origin_layer: layer3/block4/sub2/sub2_conv/Conv2D
        fread(Conv2D_23_w, sizeof(float),
              Conv2D_23->getKernelHeight() * Conv2D_23->getKernelWidth() * Conv2D_23->getOutputChannels() * Conv2D_23->getInputChannels(), fp);
        fread(Conv2D_23_b, sizeof(float), Conv2D_23->getOutputChannels(), fp);

        // origin_layer: layer3/block4/sub3/sub3_conv/Conv2D
        fread(Conv2D_24_w, sizeof(float),
              Conv2D_24->getKernelHeight() * Conv2D_24->getKernelWidth() * Conv2D_24->getOutputChannels() * Conv2D_24->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_24->getOutputChannels(); i ++) Conv2D_24_b[i] = 0.0;

        // origin_layer: layer3/block5/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_8_mean, sizeof(float), BatchNorm_8->getOutputChannels(), fp);
        fread(BatchNorm_8_variance, sizeof(float), BatchNorm_8->getOutputChannels(), fp);
        fread(BatchNorm_8_weights, sizeof(float), BatchNorm_8->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block5/sub1/sub1_conv/Conv2D
        fread(Conv2D_25_w, sizeof(float),
              Conv2D_25->getKernelHeight() * Conv2D_25->getKernelWidth() * Conv2D_25->getOutputChannels() * Conv2D_25->getInputChannels(), fp);
        fread(Conv2D_25_b, sizeof(float), Conv2D_25->getOutputChannels(), fp);

        // origin_layer: layer3/block5/sub2/sub2_conv/Conv2D
        fread(Conv2D_26_w, sizeof(float),
              Conv2D_26->getKernelHeight() * Conv2D_26->getKernelWidth() * Conv2D_26->getOutputChannels() * Conv2D_26->getInputChannels(), fp);
        fread(Conv2D_26_b, sizeof(float), Conv2D_26->getOutputChannels(), fp);

        // origin_layer: layer3/block5/sub3/sub3_conv/Conv2D
        fread(Conv2D_27_w, sizeof(float),
              Conv2D_27->getKernelHeight() * Conv2D_27->getKernelWidth() * Conv2D_27->getOutputChannels() * Conv2D_27->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_27->getOutputChannels(); i ++) Conv2D_27_b[i] = 0.0;

        // origin_layer: layer3/block6/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_9_mean, sizeof(float), BatchNorm_9->getOutputChannels(), fp);
        fread(BatchNorm_9_variance, sizeof(float), BatchNorm_9->getOutputChannels(), fp);
        fread(BatchNorm_9_weights, sizeof(float), BatchNorm_9->getOutputChannels() * 2, fp);

        // origin_layer: layer3/block6/sub1/sub1_conv/Conv2D
        fread(Conv2D_28_w, sizeof(float),
              Conv2D_28->getKernelHeight() * Conv2D_28->getKernelWidth() * Conv2D_28->getOutputChannels() * Conv2D_28->getInputChannels(), fp);
        fread(Conv2D_28_b, sizeof(float), Conv2D_28->getOutputChannels(), fp);

        // origin_layer: layer3/block6/sub2/sub2_conv/Conv2D
        fread(Conv2D_29_w, sizeof(float),
              Conv2D_29->getKernelHeight() * Conv2D_29->getKernelWidth() * Conv2D_29->getOutputChannels() * Conv2D_29->getInputChannels(), fp);
        fread(Conv2D_29_b, sizeof(float), Conv2D_29->getOutputChannels(), fp);

        // origin_layer: layer3/block6/sub3/sub3_conv/Conv2D
        fread(Conv2D_30_w, sizeof(float),
              Conv2D_30->getKernelHeight() * Conv2D_30->getKernelWidth() * Conv2D_30->getOutputChannels() * Conv2D_30->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_30->getOutputChannels(); i ++) Conv2D_30_b[i] = 0.0;

        // origin_layer: layer4/block0/common_bn_relu/FusedBatchNorm
        fread(BatchNorm_10_mean, sizeof(float), BatchNorm_10->getOutputChannels(), fp);
        fread(BatchNorm_10_variance, sizeof(float), BatchNorm_10->getOutputChannels(), fp);
        fread(BatchNorm_10_weights, sizeof(float), BatchNorm_10->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block0/sub1/sub1_conv/Conv2D
        fread(Conv2D_31_w, sizeof(float),
              Conv2D_31->getKernelHeight() * Conv2D_31->getKernelWidth() * Conv2D_31->getOutputChannels() * Conv2D_31->getInputChannels(), fp);
        fread(Conv2D_31_b, sizeof(float), Conv2D_31->getOutputChannels(), fp);

        // origin_layer: layer4/block0/sub2/sub2_conv/Conv2D
        fread(Conv2D_32_w, sizeof(float),
              Conv2D_32->getKernelHeight() * Conv2D_32->getKernelWidth() * Conv2D_32->getOutputChannels() * Conv2D_32->getInputChannels(), fp);
        fread(Conv2D_32_b, sizeof(float), Conv2D_32->getOutputChannels(), fp);

        // origin_layer: layer4/block0/sub3/sub3_conv/Conv2D
        fread(Conv2D_33_w, sizeof(float),
              Conv2D_33->getKernelHeight() * Conv2D_33->getKernelWidth() * Conv2D_33->getOutputChannels() * Conv2D_33->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_33->getOutputChannels(); i ++) Conv2D_33_b[i] = 0.0;

        // origin_layer: layer4/block0/shortcut/sub_sc/Conv2D
        fread(Conv2D_34_w, sizeof(float),
              Conv2D_34->getKernelHeight() * Conv2D_34->getKernelWidth() * Conv2D_34->getOutputChannels() * Conv2D_34->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_34->getOutputChannels(); i ++) Conv2D_34_b[i] = 0.0;

        // origin_layer: layer4/block1/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_11_mean, sizeof(float), BatchNorm_11->getOutputChannels(), fp);
        fread(BatchNorm_11_variance, sizeof(float), BatchNorm_11->getOutputChannels(), fp);
        fread(BatchNorm_11_weights, sizeof(float), BatchNorm_11->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block1/sub1/sub1_conv/Conv2D
        fread(Conv2D_35_w, sizeof(float),
              Conv2D_35->getKernelHeight() * Conv2D_35->getKernelWidth() * Conv2D_35->getOutputChannels() * Conv2D_35->getInputChannels(), fp);
        fread(Conv2D_35_b, sizeof(float), Conv2D_35->getOutputChannels(), fp);

        // origin_layer: layer4/block1/sub2/sub2_conv/Conv2D
        fread(Conv2D_36_w, sizeof(float),
              Conv2D_36->getKernelHeight() * Conv2D_36->getKernelWidth() * Conv2D_36->getOutputChannels() * Conv2D_36->getInputChannels(), fp);
        fread(Conv2D_36_b, sizeof(float), Conv2D_36->getOutputChannels(), fp);

        // origin_layer: layer4/block1/sub3/sub3_conv/Conv2D
        fread(Conv2D_37_w, sizeof(float),
              Conv2D_37->getKernelHeight() * Conv2D_37->getKernelWidth() * Conv2D_37->getOutputChannels() * Conv2D_37->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_37->getOutputChannels(); i ++) Conv2D_37_b[i] = 0.0;

        // origin_layer: layer4/block2/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_12_mean, sizeof(float), BatchNorm_12->getOutputChannels(), fp);
        fread(BatchNorm_12_variance, sizeof(float), BatchNorm_12->getOutputChannels(), fp);
        fread(BatchNorm_12_weights, sizeof(float), BatchNorm_12->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block2/sub1/sub1_conv/Conv2D
        fread(Conv2D_38_w, sizeof(float),
              Conv2D_38->getKernelHeight() * Conv2D_38->getKernelWidth() * Conv2D_38->getOutputChannels() * Conv2D_38->getInputChannels(), fp);
        fread(Conv2D_38_b, sizeof(float), Conv2D_38->getOutputChannels(), fp);

        // origin_layer: layer4/block2/sub2/sub2_conv/Conv2D
        fread(Conv2D_39_w, sizeof(float),
              Conv2D_39->getKernelHeight() * Conv2D_39->getKernelWidth() * Conv2D_39->getOutputChannels() * Conv2D_39->getInputChannels(), fp);
        fread(Conv2D_39_b, sizeof(float), Conv2D_39->getOutputChannels(), fp);

        // origin_layer: layer4/block2/sub3/sub3_conv/Conv2D
        fread(Conv2D_40_w, sizeof(float),
              Conv2D_40->getKernelHeight() * Conv2D_40->getKernelWidth() * Conv2D_40->getOutputChannels() * Conv2D_40->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_40->getOutputChannels(); i ++) Conv2D_40_b[i] = 0.0;

        // origin_layer: layer4/block3/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_13_mean, sizeof(float), BatchNorm_13->getOutputChannels(), fp);
        fread(BatchNorm_13_variance, sizeof(float), BatchNorm_13->getOutputChannels(), fp);
        fread(BatchNorm_13_weights, sizeof(float), BatchNorm_13->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block3/sub1/sub1_conv/Conv2D
        fread(Conv2D_41_w, sizeof(float),
              Conv2D_41->getKernelHeight() * Conv2D_41->getKernelWidth() * Conv2D_41->getOutputChannels() * Conv2D_41->getInputChannels(), fp);
        fread(Conv2D_41_b, sizeof(float), Conv2D_41->getOutputChannels(), fp);

        // origin_layer: layer4/block3/sub2/sub2_conv/Conv2D
        fread(Conv2D_42_w, sizeof(float),
              Conv2D_42->getKernelHeight() * Conv2D_42->getKernelWidth() * Conv2D_42->getOutputChannels() * Conv2D_42->getInputChannels(), fp);
        fread(Conv2D_42_b, sizeof(float), Conv2D_42->getOutputChannels(), fp);

        // origin_layer: layer4/block3/sub3/sub3_conv/Conv2D
        fread(Conv2D_43_w, sizeof(float),
              Conv2D_43->getKernelHeight() * Conv2D_43->getKernelWidth() * Conv2D_43->getOutputChannels() * Conv2D_43->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_43->getOutputChannels(); i ++) Conv2D_43_b[i] = 0.0;

        // origin_layer: layer4/block4/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_14_mean, sizeof(float), BatchNorm_14->getOutputChannels(), fp);
        fread(BatchNorm_14_variance, sizeof(float), BatchNorm_14->getOutputChannels(), fp);
        fread(BatchNorm_14_weights, sizeof(float), BatchNorm_14->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block4/sub1/sub1_conv/Conv2D
        fread(Conv2D_44_w, sizeof(float),
              Conv2D_44->getKernelHeight() * Conv2D_44->getKernelWidth() * Conv2D_44->getOutputChannels() * Conv2D_44->getInputChannels(), fp);
        fread(Conv2D_44_b, sizeof(float), Conv2D_44->getOutputChannels(), fp);

        // origin_layer: layer4/block4/sub2/sub2_conv/Conv2D
        fread(Conv2D_45_w, sizeof(float),
              Conv2D_45->getKernelHeight() * Conv2D_45->getKernelWidth() * Conv2D_45->getOutputChannels() * Conv2D_45->getInputChannels(), fp);
        fread(Conv2D_45_b, sizeof(float), Conv2D_45->getOutputChannels(), fp);

        // origin_layer: layer4/block4/sub3/sub3_conv/Conv2D
        fread(Conv2D_46_w, sizeof(float),
              Conv2D_46->getKernelHeight() * Conv2D_46->getKernelWidth() * Conv2D_46->getOutputChannels() * Conv2D_46->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_46->getOutputChannels(); i ++) Conv2D_46_b[i] = 0.0;

        // origin_layer: layer4/block5/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_15_mean, sizeof(float), BatchNorm_15->getOutputChannels(), fp);
        fread(BatchNorm_15_variance, sizeof(float), BatchNorm_15->getOutputChannels(), fp);
        fread(BatchNorm_15_weights, sizeof(float), BatchNorm_15->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block5/sub1/sub1_conv/Conv2D
        fread(Conv2D_47_w, sizeof(float),
              Conv2D_47->getKernelHeight() * Conv2D_47->getKernelWidth() * Conv2D_47->getOutputChannels() * Conv2D_47->getInputChannels(), fp);
        fread(Conv2D_47_b, sizeof(float), Conv2D_47->getOutputChannels(), fp);

        // origin_layer: layer4/block5/sub2/sub2_conv/Conv2D
        fread(Conv2D_48_w, sizeof(float),
              Conv2D_48->getKernelHeight() * Conv2D_48->getKernelWidth() * Conv2D_48->getOutputChannels() * Conv2D_48->getInputChannels(), fp);
        fread(Conv2D_48_b, sizeof(float), Conv2D_48->getOutputChannels(), fp);

        // origin_layer: layer4/block5/sub3/sub3_conv/Conv2D
        fread(Conv2D_49_w, sizeof(float),
              Conv2D_49->getKernelHeight() * Conv2D_49->getKernelWidth() * Conv2D_49->getOutputChannels() * Conv2D_49->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_49->getOutputChannels(); i ++) Conv2D_49_b[i] = 0.0;

        // origin_layer: layer4/block6/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_16_mean, sizeof(float), BatchNorm_16->getOutputChannels(), fp);
        fread(BatchNorm_16_variance, sizeof(float), BatchNorm_16->getOutputChannels(), fp);
        fread(BatchNorm_16_weights, sizeof(float), BatchNorm_16->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block6/sub1/sub1_conv/Conv2D
        fread(Conv2D_50_w, sizeof(float),
              Conv2D_50->getKernelHeight() * Conv2D_50->getKernelWidth() * Conv2D_50->getOutputChannels() * Conv2D_50->getInputChannels(), fp);
        fread(Conv2D_50_b, sizeof(float), Conv2D_50->getOutputChannels(), fp);

        // origin_layer: layer4/block6/sub2/sub2_conv/Conv2D
        fread(Conv2D_51_w, sizeof(float),
              Conv2D_51->getKernelHeight() * Conv2D_51->getKernelWidth() * Conv2D_51->getOutputChannels() * Conv2D_51->getInputChannels(), fp);
        fread(Conv2D_51_b, sizeof(float), Conv2D_51->getOutputChannels(), fp);

        // origin_layer: layer4/block6/sub3/sub3_conv/Conv2D
        fread(Conv2D_52_w, sizeof(float),
              Conv2D_52->getKernelHeight() * Conv2D_52->getKernelWidth() * Conv2D_52->getOutputChannels() * Conv2D_52->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_52->getOutputChannels(); i ++) Conv2D_52_b[i] = 0.0;

        // origin_layer: layer4/block7/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_17_mean, sizeof(float), BatchNorm_17->getOutputChannels(), fp);
        fread(BatchNorm_17_variance, sizeof(float), BatchNorm_17->getOutputChannels(), fp);
        fread(BatchNorm_17_weights, sizeof(float), BatchNorm_17->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block7/sub1/sub1_conv/Conv2D
        fread(Conv2D_53_w, sizeof(float),
              Conv2D_53->getKernelHeight() * Conv2D_53->getKernelWidth() * Conv2D_53->getOutputChannels() * Conv2D_53->getInputChannels(), fp);
        fread(Conv2D_53_b, sizeof(float), Conv2D_53->getOutputChannels(), fp);

        // origin_layer: layer4/block7/sub2/sub2_conv/Conv2D
        fread(Conv2D_54_w, sizeof(float),
              Conv2D_54->getKernelHeight() * Conv2D_54->getKernelWidth() * Conv2D_54->getOutputChannels() * Conv2D_54->getInputChannels(), fp);
        fread(Conv2D_54_b, sizeof(float), Conv2D_54->getOutputChannels(), fp);

        // origin_layer: layer4/block7/sub3/sub3_conv/Conv2D
        fread(Conv2D_55_w, sizeof(float),
              Conv2D_55->getKernelHeight() * Conv2D_55->getKernelWidth() * Conv2D_55->getOutputChannels() * Conv2D_55->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_55->getOutputChannels(); i ++) Conv2D_55_b[i] = 0.0;

        // origin_layer: layer4/block8/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_18_mean, sizeof(float), BatchNorm_18->getOutputChannels(), fp);
        fread(BatchNorm_18_variance, sizeof(float), BatchNorm_18->getOutputChannels(), fp);
        fread(BatchNorm_18_weights, sizeof(float), BatchNorm_18->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block8/sub1/sub1_conv/Conv2D
        fread(Conv2D_56_w, sizeof(float),
              Conv2D_56->getKernelHeight() * Conv2D_56->getKernelWidth() * Conv2D_56->getOutputChannels() * Conv2D_56->getInputChannels(), fp);
        fread(Conv2D_56_b, sizeof(float), Conv2D_56->getOutputChannels(), fp);

        // origin_layer: layer4/block8/sub2/sub2_conv/Conv2D
        fread(Conv2D_57_w, sizeof(float),
              Conv2D_57->getKernelHeight() * Conv2D_57->getKernelWidth() * Conv2D_57->getOutputChannels() * Conv2D_57->getInputChannels(), fp);
        fread(Conv2D_57_b, sizeof(float), Conv2D_57->getOutputChannels(), fp);

        // origin_layer: layer4/block8/sub3/sub3_conv/Conv2D
        fread(Conv2D_58_w, sizeof(float),
              Conv2D_58->getKernelHeight() * Conv2D_58->getKernelWidth() * Conv2D_58->getOutputChannels() * Conv2D_58->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_58->getOutputChannels(); i ++) Conv2D_58_b[i] = 0.0;

        // origin_layer: layer4/block9/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_19_mean, sizeof(float), BatchNorm_19->getOutputChannels(), fp);
        fread(BatchNorm_19_variance, sizeof(float), BatchNorm_19->getOutputChannels(), fp);
        fread(BatchNorm_19_weights, sizeof(float), BatchNorm_19->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block9/sub1/sub1_conv/Conv2D
        fread(Conv2D_59_w, sizeof(float),
              Conv2D_59->getKernelHeight() * Conv2D_59->getKernelWidth() * Conv2D_59->getOutputChannels() * Conv2D_59->getInputChannels(), fp);
        fread(Conv2D_59_b, sizeof(float), Conv2D_59->getOutputChannels(), fp);

        // origin_layer: layer4/block9/sub2/sub2_conv/Conv2D
        fread(Conv2D_60_w, sizeof(float),
              Conv2D_60->getKernelHeight() * Conv2D_60->getKernelWidth() * Conv2D_60->getOutputChannels() * Conv2D_60->getInputChannels(), fp);
        fread(Conv2D_60_b, sizeof(float), Conv2D_60->getOutputChannels(), fp);

        // origin_layer: layer4/block9/sub3/sub3_conv/Conv2D
        fread(Conv2D_61_w, sizeof(float),
              Conv2D_61->getKernelHeight() * Conv2D_61->getKernelWidth() * Conv2D_61->getOutputChannels() * Conv2D_61->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_61->getOutputChannels(); i ++) Conv2D_61_b[i] = 0.0;

        // origin_layer: layer4/block10/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_20_mean, sizeof(float), BatchNorm_20->getOutputChannels(), fp);
        fread(BatchNorm_20_variance, sizeof(float), BatchNorm_20->getOutputChannels(), fp);
        fread(BatchNorm_20_weights, sizeof(float), BatchNorm_20->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block10/sub1/sub1_conv/Conv2D
        fread(Conv2D_62_w, sizeof(float),
              Conv2D_62->getKernelHeight() * Conv2D_62->getKernelWidth() * Conv2D_62->getOutputChannels() * Conv2D_62->getInputChannels(), fp);
        fread(Conv2D_62_b, sizeof(float), Conv2D_62->getOutputChannels(), fp);

        // origin_layer: layer4/block10/sub2/sub2_conv/Conv2D
        fread(Conv2D_63_w, sizeof(float),
              Conv2D_63->getKernelHeight() * Conv2D_63->getKernelWidth() * Conv2D_63->getOutputChannels() * Conv2D_63->getInputChannels(), fp);
        fread(Conv2D_63_b, sizeof(float), Conv2D_63->getOutputChannels(), fp);

        // origin_layer: layer4/block10/sub3/sub3_conv/Conv2D
        fread(Conv2D_64_w, sizeof(float),
              Conv2D_64->getKernelHeight() * Conv2D_64->getKernelWidth() * Conv2D_64->getOutputChannels() * Conv2D_64->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_64->getOutputChannels(); i ++) Conv2D_64_b[i] = 0.0;

        // origin_layer: layer4/block11/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_21_mean, sizeof(float), BatchNorm_21->getOutputChannels(), fp);
        fread(BatchNorm_21_variance, sizeof(float), BatchNorm_21->getOutputChannels(), fp);
        fread(BatchNorm_21_weights, sizeof(float), BatchNorm_21->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block11/sub1/sub1_conv/Conv2D
        fread(Conv2D_65_w, sizeof(float),
              Conv2D_65->getKernelHeight() * Conv2D_65->getKernelWidth() * Conv2D_65->getOutputChannels() * Conv2D_65->getInputChannels(), fp);
        fread(Conv2D_65_b, sizeof(float), Conv2D_65->getOutputChannels(), fp);

        // origin_layer: layer4/block11/sub2/sub2_conv/Conv2D
        fread(Conv2D_66_w, sizeof(float),
              Conv2D_66->getKernelHeight() * Conv2D_66->getKernelWidth() * Conv2D_66->getOutputChannels() * Conv2D_66->getInputChannels(), fp);
        fread(Conv2D_66_b, sizeof(float), Conv2D_66->getOutputChannels(), fp);

        // origin_layer: layer4/block11/sub3/sub3_conv/Conv2D
        fread(Conv2D_67_w, sizeof(float),
              Conv2D_67->getKernelHeight() * Conv2D_67->getKernelWidth() * Conv2D_67->getOutputChannels() * Conv2D_67->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_67->getOutputChannels(); i ++) Conv2D_67_b[i] = 0.0;

        // origin_layer: layer4/block12/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_22_mean, sizeof(float), BatchNorm_22->getOutputChannels(), fp);
        fread(BatchNorm_22_variance, sizeof(float), BatchNorm_22->getOutputChannels(), fp);
        fread(BatchNorm_22_weights, sizeof(float), BatchNorm_22->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block12/sub1/sub1_conv/Conv2D
        fread(Conv2D_68_w, sizeof(float),
              Conv2D_68->getKernelHeight() * Conv2D_68->getKernelWidth() * Conv2D_68->getOutputChannels() * Conv2D_68->getInputChannels(), fp);
        fread(Conv2D_68_b, sizeof(float), Conv2D_68->getOutputChannels(), fp);

        // origin_layer: layer4/block12/sub2/sub2_conv/Conv2D
        fread(Conv2D_69_w, sizeof(float),
              Conv2D_69->getKernelHeight() * Conv2D_69->getKernelWidth() * Conv2D_69->getOutputChannels() * Conv2D_69->getInputChannels(), fp);
        fread(Conv2D_69_b, sizeof(float), Conv2D_69->getOutputChannels(), fp);

        // origin_layer: layer4/block12/sub3/sub3_conv/Conv2D
        fread(Conv2D_70_w, sizeof(float),
              Conv2D_70->getKernelHeight() * Conv2D_70->getKernelWidth() * Conv2D_70->getOutputChannels() * Conv2D_70->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_70->getOutputChannels(); i ++) Conv2D_70_b[i] = 0.0;

        // origin_layer: layer4/block13/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_23_mean, sizeof(float), BatchNorm_23->getOutputChannels(), fp);
        fread(BatchNorm_23_variance, sizeof(float), BatchNorm_23->getOutputChannels(), fp);
        fread(BatchNorm_23_weights, sizeof(float), BatchNorm_23->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block13/sub1/sub1_conv/Conv2D
        fread(Conv2D_71_w, sizeof(float),
              Conv2D_71->getKernelHeight() * Conv2D_71->getKernelWidth() * Conv2D_71->getOutputChannels() * Conv2D_71->getInputChannels(), fp);
        fread(Conv2D_71_b, sizeof(float), Conv2D_71->getOutputChannels(), fp);

        // origin_layer: layer4/block13/sub2/sub2_conv/Conv2D
        fread(Conv2D_72_w, sizeof(float),
              Conv2D_72->getKernelHeight() * Conv2D_72->getKernelWidth() * Conv2D_72->getOutputChannels() * Conv2D_72->getInputChannels(), fp);
        fread(Conv2D_72_b, sizeof(float), Conv2D_72->getOutputChannels(), fp);

        // origin_layer: layer4/block13/sub3/sub3_conv/Conv2D
        fread(Conv2D_73_w, sizeof(float),
              Conv2D_73->getKernelHeight() * Conv2D_73->getKernelWidth() * Conv2D_73->getOutputChannels() * Conv2D_73->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_73->getOutputChannels(); i ++) Conv2D_73_b[i] = 0.0;

        // origin_layer: layer4/block14/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_24_mean, sizeof(float), BatchNorm_24->getOutputChannels(), fp);
        fread(BatchNorm_24_variance, sizeof(float), BatchNorm_24->getOutputChannels(), fp);
        fread(BatchNorm_24_weights, sizeof(float), BatchNorm_24->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block14/sub1/sub1_conv/Conv2D
        fread(Conv2D_74_w, sizeof(float),
              Conv2D_74->getKernelHeight() * Conv2D_74->getKernelWidth() * Conv2D_74->getOutputChannels() * Conv2D_74->getInputChannels(), fp);
        fread(Conv2D_74_b, sizeof(float), Conv2D_74->getOutputChannels(), fp);

        // origin_layer: layer4/block14/sub2/sub2_conv/Conv2D
        fread(Conv2D_75_w, sizeof(float),
              Conv2D_75->getKernelHeight() * Conv2D_75->getKernelWidth() * Conv2D_75->getOutputChannels() * Conv2D_75->getInputChannels(), fp);
        fread(Conv2D_75_b, sizeof(float), Conv2D_75->getOutputChannels(), fp);

        // origin_layer: layer4/block14/sub3/sub3_conv/Conv2D
        fread(Conv2D_76_w, sizeof(float),
              Conv2D_76->getKernelHeight() * Conv2D_76->getKernelWidth() * Conv2D_76->getOutputChannels() * Conv2D_76->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_76->getOutputChannels(); i ++) Conv2D_76_b[i] = 0.0;

        // origin_layer: layer4/block15/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_25_mean, sizeof(float), BatchNorm_25->getOutputChannels(), fp);
        fread(BatchNorm_25_variance, sizeof(float), BatchNorm_25->getOutputChannels(), fp);
        fread(BatchNorm_25_weights, sizeof(float), BatchNorm_25->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block15/sub1/sub1_conv/Conv2D
        fread(Conv2D_77_w, sizeof(float),
              Conv2D_77->getKernelHeight() * Conv2D_77->getKernelWidth() * Conv2D_77->getOutputChannels() * Conv2D_77->getInputChannels(), fp);
        fread(Conv2D_77_b, sizeof(float), Conv2D_77->getOutputChannels(), fp);

        // origin_layer: layer4/block15/sub2/sub2_conv/Conv2D
        fread(Conv2D_78_w, sizeof(float),
              Conv2D_78->getKernelHeight() * Conv2D_78->getKernelWidth() * Conv2D_78->getOutputChannels() * Conv2D_78->getInputChannels(), fp);
        fread(Conv2D_78_b, sizeof(float), Conv2D_78->getOutputChannels(), fp);

        // origin_layer: layer4/block15/sub3/sub3_conv/Conv2D
        fread(Conv2D_79_w, sizeof(float),
              Conv2D_79->getKernelHeight() * Conv2D_79->getKernelWidth() * Conv2D_79->getOutputChannels() * Conv2D_79->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_79->getOutputChannels(); i ++) Conv2D_79_b[i] = 0.0;

        // origin_layer: layer4/block16/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_26_mean, sizeof(float), BatchNorm_26->getOutputChannels(), fp);
        fread(BatchNorm_26_variance, sizeof(float), BatchNorm_26->getOutputChannels(), fp);
        fread(BatchNorm_26_weights, sizeof(float), BatchNorm_26->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block16/sub1/sub1_conv/Conv2D
        fread(Conv2D_80_w, sizeof(float),
              Conv2D_80->getKernelHeight() * Conv2D_80->getKernelWidth() * Conv2D_80->getOutputChannels() * Conv2D_80->getInputChannels(), fp);
        fread(Conv2D_80_b, sizeof(float), Conv2D_80->getOutputChannels(), fp);

        // origin_layer: layer4/block16/sub2/sub2_conv/Conv2D
        fread(Conv2D_81_w, sizeof(float),
              Conv2D_81->getKernelHeight() * Conv2D_81->getKernelWidth() * Conv2D_81->getOutputChannels() * Conv2D_81->getInputChannels(), fp);
        fread(Conv2D_81_b, sizeof(float), Conv2D_81->getOutputChannels(), fp);

        // origin_layer: layer4/block16/sub3/sub3_conv/Conv2D
        fread(Conv2D_82_w, sizeof(float),
              Conv2D_82->getKernelHeight() * Conv2D_82->getKernelWidth() * Conv2D_82->getOutputChannels() * Conv2D_82->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_82->getOutputChannels(); i ++) Conv2D_82_b[i] = 0.0;

        // origin_layer: layer4/block17/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_27_mean, sizeof(float), BatchNorm_27->getOutputChannels(), fp);
        fread(BatchNorm_27_variance, sizeof(float), BatchNorm_27->getOutputChannels(), fp);
        fread(BatchNorm_27_weights, sizeof(float), BatchNorm_27->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block17/sub1/sub1_conv/Conv2D
        fread(Conv2D_83_w, sizeof(float),
              Conv2D_83->getKernelHeight() * Conv2D_83->getKernelWidth() * Conv2D_83->getOutputChannels() * Conv2D_83->getInputChannels(), fp);
        fread(Conv2D_83_b, sizeof(float), Conv2D_83->getOutputChannels(), fp);

        // origin_layer: layer4/block17/sub2/sub2_conv/Conv2D
        fread(Conv2D_84_w, sizeof(float),
              Conv2D_84->getKernelHeight() * Conv2D_84->getKernelWidth() * Conv2D_84->getOutputChannels() * Conv2D_84->getInputChannels(), fp);
        fread(Conv2D_84_b, sizeof(float), Conv2D_84->getOutputChannels(), fp);

        // origin_layer: layer4/block17/sub3/sub3_conv/Conv2D
        fread(Conv2D_85_w, sizeof(float),
              Conv2D_85->getKernelHeight() * Conv2D_85->getKernelWidth() * Conv2D_85->getOutputChannels() * Conv2D_85->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_85->getOutputChannels(); i ++) Conv2D_85_b[i] = 0.0;

        // origin_layer: layer4/block18/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_28_mean, sizeof(float), BatchNorm_28->getOutputChannels(), fp);
        fread(BatchNorm_28_variance, sizeof(float), BatchNorm_28->getOutputChannels(), fp);
        fread(BatchNorm_28_weights, sizeof(float), BatchNorm_28->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block18/sub1/sub1_conv/Conv2D
        fread(Conv2D_86_w, sizeof(float),
              Conv2D_86->getKernelHeight() * Conv2D_86->getKernelWidth() * Conv2D_86->getOutputChannels() * Conv2D_86->getInputChannels(), fp);
        fread(Conv2D_86_b, sizeof(float), Conv2D_86->getOutputChannels(), fp);

        // origin_layer: layer4/block18/sub2/sub2_conv/Conv2D
        fread(Conv2D_87_w, sizeof(float),
              Conv2D_87->getKernelHeight() * Conv2D_87->getKernelWidth() * Conv2D_87->getOutputChannels() * Conv2D_87->getInputChannels(), fp);
        fread(Conv2D_87_b, sizeof(float), Conv2D_87->getOutputChannels(), fp);

        // origin_layer: layer4/block18/sub3/sub3_conv/Conv2D
        fread(Conv2D_88_w, sizeof(float),
              Conv2D_88->getKernelHeight() * Conv2D_88->getKernelWidth() * Conv2D_88->getOutputChannels() * Conv2D_88->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_88->getOutputChannels(); i ++) Conv2D_88_b[i] = 0.0;

        // origin_layer: layer4/block19/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_29_mean, sizeof(float), BatchNorm_29->getOutputChannels(), fp);
        fread(BatchNorm_29_variance, sizeof(float), BatchNorm_29->getOutputChannels(), fp);
        fread(BatchNorm_29_weights, sizeof(float), BatchNorm_29->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block19/sub1/sub1_conv/Conv2D
        fread(Conv2D_89_w, sizeof(float),
              Conv2D_89->getKernelHeight() * Conv2D_89->getKernelWidth() * Conv2D_89->getOutputChannels() * Conv2D_89->getInputChannels(), fp);
        fread(Conv2D_89_b, sizeof(float), Conv2D_89->getOutputChannels(), fp);

        // origin_layer: layer4/block19/sub2/sub2_conv/Conv2D
        fread(Conv2D_90_w, sizeof(float),
              Conv2D_90->getKernelHeight() * Conv2D_90->getKernelWidth() * Conv2D_90->getOutputChannels() * Conv2D_90->getInputChannels(), fp);
        fread(Conv2D_90_b, sizeof(float), Conv2D_90->getOutputChannels(), fp);

        // origin_layer: layer4/block19/sub3/sub3_conv/Conv2D
        fread(Conv2D_91_w, sizeof(float),
              Conv2D_91->getKernelHeight() * Conv2D_91->getKernelWidth() * Conv2D_91->getOutputChannels() * Conv2D_91->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_91->getOutputChannels(); i ++) Conv2D_91_b[i] = 0.0;

        // origin_layer: layer4/block20/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_30_mean, sizeof(float), BatchNorm_30->getOutputChannels(), fp);
        fread(BatchNorm_30_variance, sizeof(float), BatchNorm_30->getOutputChannels(), fp);
        fread(BatchNorm_30_weights, sizeof(float), BatchNorm_30->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block20/sub1/sub1_conv/Conv2D
        fread(Conv2D_92_w, sizeof(float),
              Conv2D_92->getKernelHeight() * Conv2D_92->getKernelWidth() * Conv2D_92->getOutputChannels() * Conv2D_92->getInputChannels(), fp);
        fread(Conv2D_92_b, sizeof(float), Conv2D_92->getOutputChannels(), fp);

        // origin_layer: layer4/block20/sub2/sub2_conv/Conv2D
        fread(Conv2D_93_w, sizeof(float),
              Conv2D_93->getKernelHeight() * Conv2D_93->getKernelWidth() * Conv2D_93->getOutputChannels() * Conv2D_93->getInputChannels(), fp);
        fread(Conv2D_93_b, sizeof(float), Conv2D_93->getOutputChannels(), fp);

        // origin_layer: layer4/block20/sub3/sub3_conv/Conv2D
        fread(Conv2D_94_w, sizeof(float),
              Conv2D_94->getKernelHeight() * Conv2D_94->getKernelWidth() * Conv2D_94->getOutputChannels() * Conv2D_94->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_94->getOutputChannels(); i ++) Conv2D_94_b[i] = 0.0;

        // origin_layer: layer4/block21/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_31_mean, sizeof(float), BatchNorm_31->getOutputChannels(), fp);
        fread(BatchNorm_31_variance, sizeof(float), BatchNorm_31->getOutputChannels(), fp);
        fread(BatchNorm_31_weights, sizeof(float), BatchNorm_31->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block21/sub1/sub1_conv/Conv2D
        fread(Conv2D_95_w, sizeof(float),
              Conv2D_95->getKernelHeight() * Conv2D_95->getKernelWidth() * Conv2D_95->getOutputChannels() * Conv2D_95->getInputChannels(), fp);
        fread(Conv2D_95_b, sizeof(float), Conv2D_95->getOutputChannels(), fp);

        // origin_layer: layer4/block21/sub2/sub2_conv/Conv2D
        fread(Conv2D_96_w, sizeof(float),
              Conv2D_96->getKernelHeight() * Conv2D_96->getKernelWidth() * Conv2D_96->getOutputChannels() * Conv2D_96->getInputChannels(), fp);
        fread(Conv2D_96_b, sizeof(float), Conv2D_96->getOutputChannels(), fp);

        // origin_layer: layer4/block21/sub3/sub3_conv/Conv2D
        fread(Conv2D_97_w, sizeof(float),
              Conv2D_97->getKernelHeight() * Conv2D_97->getKernelWidth() * Conv2D_97->getOutputChannels() * Conv2D_97->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_97->getOutputChannels(); i ++) Conv2D_97_b[i] = 0.0;

        // origin_layer: layer4/block22/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_32_mean, sizeof(float), BatchNorm_32->getOutputChannels(), fp);
        fread(BatchNorm_32_variance, sizeof(float), BatchNorm_32->getOutputChannels(), fp);
        fread(BatchNorm_32_weights, sizeof(float), BatchNorm_32->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block22/sub1/sub1_conv/Conv2D
        fread(Conv2D_98_w, sizeof(float),
              Conv2D_98->getKernelHeight() * Conv2D_98->getKernelWidth() * Conv2D_98->getOutputChannels() * Conv2D_98->getInputChannels(), fp);
        fread(Conv2D_98_b, sizeof(float), Conv2D_98->getOutputChannels(), fp);

        // origin_layer: layer4/block22/sub2/sub2_conv/Conv2D
        fread(Conv2D_99_w, sizeof(float),
              Conv2D_99->getKernelHeight() * Conv2D_99->getKernelWidth() * Conv2D_99->getOutputChannels() * Conv2D_99->getInputChannels(), fp);
        fread(Conv2D_99_b, sizeof(float), Conv2D_99->getOutputChannels(), fp);

        // origin_layer: layer4/block22/sub3/sub3_conv/Conv2D
        fread(Conv2D_100_w, sizeof(float),
              Conv2D_100->getKernelHeight() * Conv2D_100->getKernelWidth() * Conv2D_100->getOutputChannels() * Conv2D_100->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_100->getOutputChannels(); i ++) Conv2D_100_b[i] = 0.0;

        // origin_layer: layer4/block23/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_33_mean, sizeof(float), BatchNorm_33->getOutputChannels(), fp);
        fread(BatchNorm_33_variance, sizeof(float), BatchNorm_33->getOutputChannels(), fp);
        fread(BatchNorm_33_weights, sizeof(float), BatchNorm_33->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block23/sub1/sub1_conv/Conv2D
        fread(Conv2D_101_w, sizeof(float),
              Conv2D_101->getKernelHeight() * Conv2D_101->getKernelWidth() * Conv2D_101->getOutputChannels() * Conv2D_101->getInputChannels(), fp);
        fread(Conv2D_101_b, sizeof(float), Conv2D_101->getOutputChannels(), fp);

        // origin_layer: layer4/block23/sub2/sub2_conv/Conv2D
        fread(Conv2D_102_w, sizeof(float),
              Conv2D_102->getKernelHeight() * Conv2D_102->getKernelWidth() * Conv2D_102->getOutputChannels() * Conv2D_102->getInputChannels(), fp);
        fread(Conv2D_102_b, sizeof(float), Conv2D_102->getOutputChannels(), fp);

        // origin_layer: layer4/block23/sub3/sub3_conv/Conv2D
        fread(Conv2D_103_w, sizeof(float),
              Conv2D_103->getKernelHeight() * Conv2D_103->getKernelWidth() * Conv2D_103->getOutputChannels() * Conv2D_103->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_103->getOutputChannels(); i ++) Conv2D_103_b[i] = 0.0;

        // origin_layer: layer4/block24/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_34_mean, sizeof(float), BatchNorm_34->getOutputChannels(), fp);
        fread(BatchNorm_34_variance, sizeof(float), BatchNorm_34->getOutputChannels(), fp);
        fread(BatchNorm_34_weights, sizeof(float), BatchNorm_34->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block24/sub1/sub1_conv/Conv2D
        fread(Conv2D_104_w, sizeof(float),
              Conv2D_104->getKernelHeight() * Conv2D_104->getKernelWidth() * Conv2D_104->getOutputChannels() * Conv2D_104->getInputChannels(), fp);
        fread(Conv2D_104_b, sizeof(float), Conv2D_104->getOutputChannels(), fp);

        // origin_layer: layer4/block24/sub2/sub2_conv/Conv2D
        fread(Conv2D_105_w, sizeof(float),
              Conv2D_105->getKernelHeight() * Conv2D_105->getKernelWidth() * Conv2D_105->getOutputChannels() * Conv2D_105->getInputChannels(), fp);
        fread(Conv2D_105_b, sizeof(float), Conv2D_105->getOutputChannels(), fp);

        // origin_layer: layer4/block24/sub3/sub3_conv/Conv2D
        fread(Conv2D_106_w, sizeof(float),
              Conv2D_106->getKernelHeight() * Conv2D_106->getKernelWidth() * Conv2D_106->getOutputChannels() * Conv2D_106->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_106->getOutputChannels(); i ++) Conv2D_106_b[i] = 0.0;

        // origin_layer: layer4/block25/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_35_mean, sizeof(float), BatchNorm_35->getOutputChannels(), fp);
        fread(BatchNorm_35_variance, sizeof(float), BatchNorm_35->getOutputChannels(), fp);
        fread(BatchNorm_35_weights, sizeof(float), BatchNorm_35->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block25/sub1/sub1_conv/Conv2D
        fread(Conv2D_107_w, sizeof(float),
              Conv2D_107->getKernelHeight() * Conv2D_107->getKernelWidth() * Conv2D_107->getOutputChannels() * Conv2D_107->getInputChannels(), fp);
        fread(Conv2D_107_b, sizeof(float), Conv2D_107->getOutputChannels(), fp);

        // origin_layer: layer4/block25/sub2/sub2_conv/Conv2D
        fread(Conv2D_108_w, sizeof(float),
              Conv2D_108->getKernelHeight() * Conv2D_108->getKernelWidth() * Conv2D_108->getOutputChannels() * Conv2D_108->getInputChannels(), fp);
        fread(Conv2D_108_b, sizeof(float), Conv2D_108->getOutputChannels(), fp);

        // origin_layer: layer4/block25/sub3/sub3_conv/Conv2D
        fread(Conv2D_109_w, sizeof(float),
              Conv2D_109->getKernelHeight() * Conv2D_109->getKernelWidth() * Conv2D_109->getOutputChannels() * Conv2D_109->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_109->getOutputChannels(); i ++) Conv2D_109_b[i] = 0.0;

        // origin_layer: layer4/block26/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_36_mean, sizeof(float), BatchNorm_36->getOutputChannels(), fp);
        fread(BatchNorm_36_variance, sizeof(float), BatchNorm_36->getOutputChannels(), fp);
        fread(BatchNorm_36_weights, sizeof(float), BatchNorm_36->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block26/sub1/sub1_conv/Conv2D
        fread(Conv2D_110_w, sizeof(float),
              Conv2D_110->getKernelHeight() * Conv2D_110->getKernelWidth() * Conv2D_110->getOutputChannels() * Conv2D_110->getInputChannels(), fp);
        fread(Conv2D_110_b, sizeof(float), Conv2D_110->getOutputChannels(), fp);

        // origin_layer: layer4/block26/sub2/sub2_conv/Conv2D
        fread(Conv2D_111_w, sizeof(float),
              Conv2D_111->getKernelHeight() * Conv2D_111->getKernelWidth() * Conv2D_111->getOutputChannels() * Conv2D_111->getInputChannels(), fp);
        fread(Conv2D_111_b, sizeof(float), Conv2D_111->getOutputChannels(), fp);

        // origin_layer: layer4/block26/sub3/sub3_conv/Conv2D
        fread(Conv2D_112_w, sizeof(float),
              Conv2D_112->getKernelHeight() * Conv2D_112->getKernelWidth() * Conv2D_112->getOutputChannels() * Conv2D_112->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_112->getOutputChannels(); i ++) Conv2D_112_b[i] = 0.0;

        // origin_layer: layer4/block27/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_37_mean, sizeof(float), BatchNorm_37->getOutputChannels(), fp);
        fread(BatchNorm_37_variance, sizeof(float), BatchNorm_37->getOutputChannels(), fp);
        fread(BatchNorm_37_weights, sizeof(float), BatchNorm_37->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block27/sub1/sub1_conv/Conv2D
        fread(Conv2D_113_w, sizeof(float),
              Conv2D_113->getKernelHeight() * Conv2D_113->getKernelWidth() * Conv2D_113->getOutputChannels() * Conv2D_113->getInputChannels(), fp);
        fread(Conv2D_113_b, sizeof(float), Conv2D_113->getOutputChannels(), fp);

        // origin_layer: layer4/block27/sub2/sub2_conv/Conv2D
        fread(Conv2D_114_w, sizeof(float),
              Conv2D_114->getKernelHeight() * Conv2D_114->getKernelWidth() * Conv2D_114->getOutputChannels() * Conv2D_114->getInputChannels(), fp);
        fread(Conv2D_114_b, sizeof(float), Conv2D_114->getOutputChannels(), fp);

        // origin_layer: layer4/block27/sub3/sub3_conv/Conv2D
        fread(Conv2D_115_w, sizeof(float),
              Conv2D_115->getKernelHeight() * Conv2D_115->getKernelWidth() * Conv2D_115->getOutputChannels() * Conv2D_115->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_115->getOutputChannels(); i ++) Conv2D_115_b[i] = 0.0;

        // origin_layer: layer4/block28/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_38_mean, sizeof(float), BatchNorm_38->getOutputChannels(), fp);
        fread(BatchNorm_38_variance, sizeof(float), BatchNorm_38->getOutputChannels(), fp);
        fread(BatchNorm_38_weights, sizeof(float), BatchNorm_38->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block28/sub1/sub1_conv/Conv2D
        fread(Conv2D_116_w, sizeof(float),
              Conv2D_116->getKernelHeight() * Conv2D_116->getKernelWidth() * Conv2D_116->getOutputChannels() * Conv2D_116->getInputChannels(), fp);
        fread(Conv2D_116_b, sizeof(float), Conv2D_116->getOutputChannels(), fp);

        // origin_layer: layer4/block28/sub2/sub2_conv/Conv2D
        fread(Conv2D_117_w, sizeof(float),
              Conv2D_117->getKernelHeight() * Conv2D_117->getKernelWidth() * Conv2D_117->getOutputChannels() * Conv2D_117->getInputChannels(), fp);
        fread(Conv2D_117_b, sizeof(float), Conv2D_117->getOutputChannels(), fp);

        // origin_layer: layer4/block28/sub3/sub3_conv/Conv2D
        fread(Conv2D_118_w, sizeof(float),
              Conv2D_118->getKernelHeight() * Conv2D_118->getKernelWidth() * Conv2D_118->getOutputChannels() * Conv2D_118->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_118->getOutputChannels(); i ++) Conv2D_118_b[i] = 0.0;

        // origin_layer: layer4/block29/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_39_mean, sizeof(float), BatchNorm_39->getOutputChannels(), fp);
        fread(BatchNorm_39_variance, sizeof(float), BatchNorm_39->getOutputChannels(), fp);
        fread(BatchNorm_39_weights, sizeof(float), BatchNorm_39->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block29/sub1/sub1_conv/Conv2D
        fread(Conv2D_119_w, sizeof(float),
              Conv2D_119->getKernelHeight() * Conv2D_119->getKernelWidth() * Conv2D_119->getOutputChannels() * Conv2D_119->getInputChannels(), fp);
        fread(Conv2D_119_b, sizeof(float), Conv2D_119->getOutputChannels(), fp);

        // origin_layer: layer4/block29/sub2/sub2_conv/Conv2D
        fread(Conv2D_120_w, sizeof(float),
              Conv2D_120->getKernelHeight() * Conv2D_120->getKernelWidth() * Conv2D_120->getOutputChannels() * Conv2D_120->getInputChannels(), fp);
        fread(Conv2D_120_b, sizeof(float), Conv2D_120->getOutputChannels(), fp);

        // origin_layer: layer4/block29/sub3/sub3_conv/Conv2D
        fread(Conv2D_121_w, sizeof(float),
              Conv2D_121->getKernelHeight() * Conv2D_121->getKernelWidth() * Conv2D_121->getOutputChannels() * Conv2D_121->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_121->getOutputChannels(); i ++) Conv2D_121_b[i] = 0.0;

        // origin_layer: layer4/block30/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_40_mean, sizeof(float), BatchNorm_40->getOutputChannels(), fp);
        fread(BatchNorm_40_variance, sizeof(float), BatchNorm_40->getOutputChannels(), fp);
        fread(BatchNorm_40_weights, sizeof(float), BatchNorm_40->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block30/sub1/sub1_conv/Conv2D
        fread(Conv2D_122_w, sizeof(float),
              Conv2D_122->getKernelHeight() * Conv2D_122->getKernelWidth() * Conv2D_122->getOutputChannels() * Conv2D_122->getInputChannels(), fp);
        fread(Conv2D_122_b, sizeof(float), Conv2D_122->getOutputChannels(), fp);

        // origin_layer: layer4/block30/sub2/sub2_conv/Conv2D
        fread(Conv2D_123_w, sizeof(float),
              Conv2D_123->getKernelHeight() * Conv2D_123->getKernelWidth() * Conv2D_123->getOutputChannels() * Conv2D_123->getInputChannels(), fp);
        fread(Conv2D_123_b, sizeof(float), Conv2D_123->getOutputChannels(), fp);

        // origin_layer: layer4/block30/sub3/sub3_conv/Conv2D
        fread(Conv2D_124_w, sizeof(float),
              Conv2D_124->getKernelHeight() * Conv2D_124->getKernelWidth() * Conv2D_124->getOutputChannels() * Conv2D_124->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_124->getOutputChannels(); i ++) Conv2D_124_b[i] = 0.0;

        // origin_layer: layer4/block31/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_41_mean, sizeof(float), BatchNorm_41->getOutputChannels(), fp);
        fread(BatchNorm_41_variance, sizeof(float), BatchNorm_41->getOutputChannels(), fp);
        fread(BatchNorm_41_weights, sizeof(float), BatchNorm_41->getOutputChannels() * 2, fp);

        // origin_layer: layer4/block31/sub1/sub1_conv/Conv2D
        fread(Conv2D_125_w, sizeof(float),
              Conv2D_125->getKernelHeight() * Conv2D_125->getKernelWidth() * Conv2D_125->getOutputChannels() * Conv2D_125->getInputChannels(), fp);
        fread(Conv2D_125_b, sizeof(float), Conv2D_125->getOutputChannels(), fp);

        // origin_layer: layer4/block31/sub2/sub2_conv/Conv2D
        fread(Conv2D_126_w, sizeof(float),
              Conv2D_126->getKernelHeight() * Conv2D_126->getKernelWidth() * Conv2D_126->getOutputChannels() * Conv2D_126->getInputChannels(), fp);
        fread(Conv2D_126_b, sizeof(float), Conv2D_126->getOutputChannels(), fp);

        // origin_layer: layer4/block31/sub3/sub3_conv/Conv2D
        fread(Conv2D_127_w, sizeof(float),
              Conv2D_127->getKernelHeight() * Conv2D_127->getKernelWidth() * Conv2D_127->getOutputChannels() * Conv2D_127->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_127->getOutputChannels(); i ++) Conv2D_127_b[i] = 0.0;

        // origin_layer: layer5/block0/common_bn_relu/FusedBatchNorm
        fread(BatchNorm_42_mean, sizeof(float), BatchNorm_42->getOutputChannels(), fp);
        fread(BatchNorm_42_variance, sizeof(float), BatchNorm_42->getOutputChannels(), fp);
        fread(BatchNorm_42_weights, sizeof(float), BatchNorm_42->getOutputChannels() * 2, fp);

        // origin_layer: layer5/block0/sub1/sub1_conv/Conv2D
        fread(Conv2D_128_w, sizeof(float),
              Conv2D_128->getKernelHeight() * Conv2D_128->getKernelWidth() * Conv2D_128->getOutputChannels() * Conv2D_128->getInputChannels(), fp);
        fread(Conv2D_128_b, sizeof(float), Conv2D_128->getOutputChannels(), fp);

        // origin_layer: layer5/block0/sub2/sub2_conv/Conv2D
        fread(Conv2D_129_w, sizeof(float),
              Conv2D_129->getKernelHeight() * Conv2D_129->getKernelWidth() * Conv2D_129->getOutputChannels() * Conv2D_129->getInputChannels(), fp);
        fread(Conv2D_129_b, sizeof(float), Conv2D_129->getOutputChannels(), fp);

        // origin_layer: layer5/block0/sub3/sub3_conv/Conv2D
        fread(Conv2D_130_w, sizeof(float),
              Conv2D_130->getKernelHeight() * Conv2D_130->getKernelWidth() * Conv2D_130->getOutputChannels() * Conv2D_130->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_130->getOutputChannels(); i ++) Conv2D_130_b[i] = 0.0;

        // origin_layer: layer5/block0/shortcut/sub_sc/Conv2D
        fread(Conv2D_131_w, sizeof(float),
              Conv2D_131->getKernelHeight() * Conv2D_131->getKernelWidth() * Conv2D_131->getOutputChannels() * Conv2D_131->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_131->getOutputChannels(); i ++) Conv2D_131_b[i] = 0.0;

        // origin_layer: layer5/block1/residual_bn_relu/FusedBatchNorm
        fread(BatchNorm_43_mean, sizeof(float), BatchNorm_43->getOutputChannels(), fp);
        fread(BatchNorm_43_variance, sizeof(float), BatchNorm_43->getOutputChannels(), fp);
        fread(BatchNorm_43_weights, sizeof(float), BatchNorm_43->getOutputChannels() * 2, fp);

        // origin_layer: layer5/block1/sub1/sub1_conv/Conv2D
        fread(Conv2D_132_w, sizeof(float),
              Conv2D_132->getKernelHeight() * Conv2D_132->getKernelWidth() * Conv2D_132->getOutputChannels() * Conv2D_132->getInputChannels(), fp);
        fread(Conv2D_132_b, sizeof(float), Conv2D_132->getOutputChannels(), fp);

        // origin_layer: layer5/block1/sub2/sub2_conv/Conv2D
        fread(Conv2D_133_w, sizeof(float),
              Conv2D_133->getKernelHeight() * Conv2D_133->getKernelWidth() * Conv2D_133->getOutputChannels() * Conv2D_133->getInputChannels(), fp);
        fread(Conv2D_133_b, sizeof(float), Conv2D_133->getOutputChannels(), fp);

        // origin_layer: layer5/block1/sub3/sub3_conv/Conv2D
        fread(Conv2D_134_w, sizeof(float),
              Conv2D_134->getKernelHeight() * Conv2D_134->getKernelWidth() * Conv2D_134->getOutputChannels() * Conv2D_134->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_134->getOutputChannels(); i ++) Conv2D_134_b[i] = 0.0;

        // origin_layer: avg_fc/fc_bn/FusedBatchNorm
        fread(BatchNorm_44_mean, sizeof(float), BatchNorm_44->getOutputChannels(), fp);
        fread(BatchNorm_44_variance, sizeof(float), BatchNorm_44->getOutputChannels(), fp);
        fread(BatchNorm_44_weights, sizeof(float), BatchNorm_44->getOutputChannels() * 2, fp);

        // origin_layer: avg_fc/fc6/Conv2D
        fread(Conv2D_135_w, sizeof(float),
              Conv2D_135->getKernelHeight() * Conv2D_135->getKernelWidth() * Conv2D_135->getOutputChannels() * Conv2D_135->getInputChannels(), fp);
        for(int i = 0; i < Conv2D_135->getOutputChannels(); i ++) Conv2D_135_b[i] = 0.0;

        fclose(fp);
    }


    void print_output() {
        float *pConv2D_1 = (float*)Conv2D_1_out->get_data_handle();
        float *pMaxPool_1 = (float*)MaxPool_1_out->get_data_handle();
        float *pBatchNorm_1 = (float*)BatchNorm_1_out->get_data_handle();
        float *pConv2D_2 = (float*)Conv2D_2_out->get_data_handle();
        float *pConv2D_3 = (float*)Conv2D_3_out->get_data_handle();
        float *pConv2D_4 = (float*)Conv2D_4_out->get_data_handle();
        float *pConv2D_5 = (float*)Conv2D_5_out->get_data_handle();
        float *pAdd_1 = (float*)Add_1_out->get_data_handle();
        float *pBatchNorm_2 = (float*)BatchNorm_2_out->get_data_handle();
        float *pConv2D_6 = (float*)Conv2D_6_out->get_data_handle();
        float *pConv2D_7 = (float*)Conv2D_7_out->get_data_handle();
        float *pConv2D_8 = (float*)Conv2D_8_out->get_data_handle();
        float *pAdd_2 = (float*)Add_2_out->get_data_handle();
        float *pBatchNorm_3 = (float*)BatchNorm_3_out->get_data_handle();
        float *pConv2D_9 = (float*)Conv2D_9_out->get_data_handle();
        float *pConv2D_10 = (float*)Conv2D_10_out->get_data_handle();
        float *pConv2D_11 = (float*)Conv2D_11_out->get_data_handle();
        float *pConv2D_12 = (float*)Conv2D_12_out->get_data_handle();
        float *pAdd_3 = (float*)Add_3_out->get_data_handle();
        float *pBatchNorm_4 = (float*)BatchNorm_4_out->get_data_handle();
        float *pConv2D_13 = (float*)Conv2D_13_out->get_data_handle();
        float *pConv2D_14 = (float*)Conv2D_14_out->get_data_handle();
        float *pConv2D_15 = (float*)Conv2D_15_out->get_data_handle();
        float *pAdd_4 = (float*)Add_4_out->get_data_handle();
        float *pBatchNorm_5 = (float*)BatchNorm_5_out->get_data_handle();
        float *pConv2D_16 = (float*)Conv2D_16_out->get_data_handle();
        float *pConv2D_17 = (float*)Conv2D_17_out->get_data_handle();
        float *pConv2D_18 = (float*)Conv2D_18_out->get_data_handle();
        float *pAdd_5 = (float*)Add_5_out->get_data_handle();
        float *pBatchNorm_6 = (float*)BatchNorm_6_out->get_data_handle();
        float *pConv2D_19 = (float*)Conv2D_19_out->get_data_handle();
        float *pConv2D_20 = (float*)Conv2D_20_out->get_data_handle();
        float *pConv2D_21 = (float*)Conv2D_21_out->get_data_handle();
        float *pAdd_6 = (float*)Add_6_out->get_data_handle();
        float *pBatchNorm_7 = (float*)BatchNorm_7_out->get_data_handle();
        float *pConv2D_22 = (float*)Conv2D_22_out->get_data_handle();
        float *pConv2D_23 = (float*)Conv2D_23_out->get_data_handle();
        float *pConv2D_24 = (float*)Conv2D_24_out->get_data_handle();
        float *pAdd_7 = (float*)Add_7_out->get_data_handle();
        float *pBatchNorm_8 = (float*)BatchNorm_8_out->get_data_handle();
        float *pConv2D_25 = (float*)Conv2D_25_out->get_data_handle();
        float *pConv2D_26 = (float*)Conv2D_26_out->get_data_handle();
        float *pConv2D_27 = (float*)Conv2D_27_out->get_data_handle();
        float *pAdd_8 = (float*)Add_8_out->get_data_handle();
        float *pBatchNorm_9 = (float*)BatchNorm_9_out->get_data_handle();
        float *pConv2D_28 = (float*)Conv2D_28_out->get_data_handle();
        float *pConv2D_29 = (float*)Conv2D_29_out->get_data_handle();
        float *pConv2D_30 = (float*)Conv2D_30_out->get_data_handle();
        float *pAdd_9 = (float*)Add_9_out->get_data_handle();
        float *pBatchNorm_10 = (float*)BatchNorm_10_out->get_data_handle();
        float *pConv2D_31 = (float*)Conv2D_31_out->get_data_handle();
        float *pConv2D_32 = (float*)Conv2D_32_out->get_data_handle();
        float *pConv2D_33 = (float*)Conv2D_33_out->get_data_handle();
        float *pConv2D_34 = (float*)Conv2D_34_out->get_data_handle();
        float *pAdd_10 = (float*)Add_10_out->get_data_handle();
        float *pBatchNorm_11 = (float*)BatchNorm_11_out->get_data_handle();
        float *pConv2D_35 = (float*)Conv2D_35_out->get_data_handle();
        float *pConv2D_36 = (float*)Conv2D_36_out->get_data_handle();
        float *pConv2D_37 = (float*)Conv2D_37_out->get_data_handle();
        float *pAdd_11 = (float*)Add_11_out->get_data_handle();
        float *pBatchNorm_12 = (float*)BatchNorm_12_out->get_data_handle();
        float *pConv2D_38 = (float*)Conv2D_38_out->get_data_handle();
        float *pConv2D_39 = (float*)Conv2D_39_out->get_data_handle();
        float *pConv2D_40 = (float*)Conv2D_40_out->get_data_handle();
        float *pAdd_12 = (float*)Add_12_out->get_data_handle();
        float *pBatchNorm_13 = (float*)BatchNorm_13_out->get_data_handle();
        float *pConv2D_41 = (float*)Conv2D_41_out->get_data_handle();
        float *pConv2D_42 = (float*)Conv2D_42_out->get_data_handle();
        float *pConv2D_43 = (float*)Conv2D_43_out->get_data_handle();
        float *pAdd_13 = (float*)Add_13_out->get_data_handle();
        float *pBatchNorm_14 = (float*)BatchNorm_14_out->get_data_handle();
        float *pConv2D_44 = (float*)Conv2D_44_out->get_data_handle();
        float *pConv2D_45 = (float*)Conv2D_45_out->get_data_handle();
        float *pConv2D_46 = (float*)Conv2D_46_out->get_data_handle();
        float *pAdd_14 = (float*)Add_14_out->get_data_handle();
        float *pBatchNorm_15 = (float*)BatchNorm_15_out->get_data_handle();
        float *pConv2D_47 = (float*)Conv2D_47_out->get_data_handle();
        float *pConv2D_48 = (float*)Conv2D_48_out->get_data_handle();
        float *pConv2D_49 = (float*)Conv2D_49_out->get_data_handle();
        float *pAdd_15 = (float*)Add_15_out->get_data_handle();
        float *pBatchNorm_16 = (float*)BatchNorm_16_out->get_data_handle();
        float *pConv2D_50 = (float*)Conv2D_50_out->get_data_handle();
        float *pConv2D_51 = (float*)Conv2D_51_out->get_data_handle();
        float *pConv2D_52 = (float*)Conv2D_52_out->get_data_handle();
        float *pAdd_16 = (float*)Add_16_out->get_data_handle();
        float *pBatchNorm_17 = (float*)BatchNorm_17_out->get_data_handle();
        float *pConv2D_53 = (float*)Conv2D_53_out->get_data_handle();
        float *pConv2D_54 = (float*)Conv2D_54_out->get_data_handle();
        float *pConv2D_55 = (float*)Conv2D_55_out->get_data_handle();
        float *pAdd_17 = (float*)Add_17_out->get_data_handle();
        float *pBatchNorm_18 = (float*)BatchNorm_18_out->get_data_handle();
        float *pConv2D_56 = (float*)Conv2D_56_out->get_data_handle();
        float *pConv2D_57 = (float*)Conv2D_57_out->get_data_handle();
        float *pConv2D_58 = (float*)Conv2D_58_out->get_data_handle();
        float *pAdd_18 = (float*)Add_18_out->get_data_handle();
        float *pBatchNorm_19 = (float*)BatchNorm_19_out->get_data_handle();
        float *pConv2D_59 = (float*)Conv2D_59_out->get_data_handle();
        float *pConv2D_60 = (float*)Conv2D_60_out->get_data_handle();
        float *pConv2D_61 = (float*)Conv2D_61_out->get_data_handle();
        float *pAdd_19 = (float*)Add_19_out->get_data_handle();
        float *pBatchNorm_20 = (float*)BatchNorm_20_out->get_data_handle();
        float *pConv2D_62 = (float*)Conv2D_62_out->get_data_handle();
        float *pConv2D_63 = (float*)Conv2D_63_out->get_data_handle();
        float *pConv2D_64 = (float*)Conv2D_64_out->get_data_handle();
        float *pAdd_20 = (float*)Add_20_out->get_data_handle();
        float *pBatchNorm_21 = (float*)BatchNorm_21_out->get_data_handle();
        float *pConv2D_65 = (float*)Conv2D_65_out->get_data_handle();
        float *pConv2D_66 = (float*)Conv2D_66_out->get_data_handle();
        float *pConv2D_67 = (float*)Conv2D_67_out->get_data_handle();
        float *pAdd_21 = (float*)Add_21_out->get_data_handle();
        float *pBatchNorm_22 = (float*)BatchNorm_22_out->get_data_handle();
        float *pConv2D_68 = (float*)Conv2D_68_out->get_data_handle();
        float *pConv2D_69 = (float*)Conv2D_69_out->get_data_handle();
        float *pConv2D_70 = (float*)Conv2D_70_out->get_data_handle();
        float *pAdd_22 = (float*)Add_22_out->get_data_handle();
        float *pBatchNorm_23 = (float*)BatchNorm_23_out->get_data_handle();
        float *pConv2D_71 = (float*)Conv2D_71_out->get_data_handle();
        float *pConv2D_72 = (float*)Conv2D_72_out->get_data_handle();
        float *pConv2D_73 = (float*)Conv2D_73_out->get_data_handle();
        float *pAdd_23 = (float*)Add_23_out->get_data_handle();
        float *pBatchNorm_24 = (float*)BatchNorm_24_out->get_data_handle();
        float *pConv2D_74 = (float*)Conv2D_74_out->get_data_handle();
        float *pConv2D_75 = (float*)Conv2D_75_out->get_data_handle();
        float *pConv2D_76 = (float*)Conv2D_76_out->get_data_handle();
        float *pAdd_24 = (float*)Add_24_out->get_data_handle();
        float *pBatchNorm_25 = (float*)BatchNorm_25_out->get_data_handle();
        float *pConv2D_77 = (float*)Conv2D_77_out->get_data_handle();
        float *pConv2D_78 = (float*)Conv2D_78_out->get_data_handle();
        float *pConv2D_79 = (float*)Conv2D_79_out->get_data_handle();
        float *pAdd_25 = (float*)Add_25_out->get_data_handle();
        float *pBatchNorm_26 = (float*)BatchNorm_26_out->get_data_handle();
        float *pConv2D_80 = (float*)Conv2D_80_out->get_data_handle();
        float *pConv2D_81 = (float*)Conv2D_81_out->get_data_handle();
        float *pConv2D_82 = (float*)Conv2D_82_out->get_data_handle();
        float *pAdd_26 = (float*)Add_26_out->get_data_handle();
        float *pBatchNorm_27 = (float*)BatchNorm_27_out->get_data_handle();
        float *pConv2D_83 = (float*)Conv2D_83_out->get_data_handle();
        float *pConv2D_84 = (float*)Conv2D_84_out->get_data_handle();
        float *pConv2D_85 = (float*)Conv2D_85_out->get_data_handle();
        float *pAdd_27 = (float*)Add_27_out->get_data_handle();
        float *pBatchNorm_28 = (float*)BatchNorm_28_out->get_data_handle();
        float *pConv2D_86 = (float*)Conv2D_86_out->get_data_handle();
        float *pConv2D_87 = (float*)Conv2D_87_out->get_data_handle();
        float *pConv2D_88 = (float*)Conv2D_88_out->get_data_handle();
        float *pAdd_28 = (float*)Add_28_out->get_data_handle();
        float *pBatchNorm_29 = (float*)BatchNorm_29_out->get_data_handle();
        float *pConv2D_89 = (float*)Conv2D_89_out->get_data_handle();
        float *pConv2D_90 = (float*)Conv2D_90_out->get_data_handle();
        float *pConv2D_91 = (float*)Conv2D_91_out->get_data_handle();
        float *pAdd_29 = (float*)Add_29_out->get_data_handle();
        float *pBatchNorm_30 = (float*)BatchNorm_30_out->get_data_handle();
        float *pConv2D_92 = (float*)Conv2D_92_out->get_data_handle();
        float *pConv2D_93 = (float*)Conv2D_93_out->get_data_handle();
        float *pConv2D_94 = (float*)Conv2D_94_out->get_data_handle();
        float *pAdd_30 = (float*)Add_30_out->get_data_handle();
        float *pBatchNorm_31 = (float*)BatchNorm_31_out->get_data_handle();
        float *pConv2D_95 = (float*)Conv2D_95_out->get_data_handle();
        float *pConv2D_96 = (float*)Conv2D_96_out->get_data_handle();
        float *pConv2D_97 = (float*)Conv2D_97_out->get_data_handle();
        float *pAdd_31 = (float*)Add_31_out->get_data_handle();
        float *pBatchNorm_32 = (float*)BatchNorm_32_out->get_data_handle();
        float *pConv2D_98 = (float*)Conv2D_98_out->get_data_handle();
        float *pConv2D_99 = (float*)Conv2D_99_out->get_data_handle();
        float *pConv2D_100 = (float*)Conv2D_100_out->get_data_handle();
        float *pAdd_32 = (float*)Add_32_out->get_data_handle();
        float *pBatchNorm_33 = (float*)BatchNorm_33_out->get_data_handle();
        float *pConv2D_101 = (float*)Conv2D_101_out->get_data_handle();
        float *pConv2D_102 = (float*)Conv2D_102_out->get_data_handle();
        float *pConv2D_103 = (float*)Conv2D_103_out->get_data_handle();
        float *pAdd_33 = (float*)Add_33_out->get_data_handle();
        float *pBatchNorm_34 = (float*)BatchNorm_34_out->get_data_handle();
        float *pConv2D_104 = (float*)Conv2D_104_out->get_data_handle();
        float *pConv2D_105 = (float*)Conv2D_105_out->get_data_handle();
        float *pConv2D_106 = (float*)Conv2D_106_out->get_data_handle();
        float *pAdd_34 = (float*)Add_34_out->get_data_handle();
        float *pBatchNorm_35 = (float*)BatchNorm_35_out->get_data_handle();
        float *pConv2D_107 = (float*)Conv2D_107_out->get_data_handle();
        float *pConv2D_108 = (float*)Conv2D_108_out->get_data_handle();
        float *pConv2D_109 = (float*)Conv2D_109_out->get_data_handle();
        float *pAdd_35 = (float*)Add_35_out->get_data_handle();
        float *pBatchNorm_36 = (float*)BatchNorm_36_out->get_data_handle();
        float *pConv2D_110 = (float*)Conv2D_110_out->get_data_handle();
        float *pConv2D_111 = (float*)Conv2D_111_out->get_data_handle();
        float *pConv2D_112 = (float*)Conv2D_112_out->get_data_handle();
        float *pAdd_36 = (float*)Add_36_out->get_data_handle();
        float *pBatchNorm_37 = (float*)BatchNorm_37_out->get_data_handle();
        float *pConv2D_113 = (float*)Conv2D_113_out->get_data_handle();
        float *pConv2D_114 = (float*)Conv2D_114_out->get_data_handle();
        float *pConv2D_115 = (float*)Conv2D_115_out->get_data_handle();
        float *pAdd_37 = (float*)Add_37_out->get_data_handle();
        float *pBatchNorm_38 = (float*)BatchNorm_38_out->get_data_handle();
        float *pConv2D_116 = (float*)Conv2D_116_out->get_data_handle();
        float *pConv2D_117 = (float*)Conv2D_117_out->get_data_handle();
        float *pConv2D_118 = (float*)Conv2D_118_out->get_data_handle();
        float *pAdd_38 = (float*)Add_38_out->get_data_handle();
        float *pBatchNorm_39 = (float*)BatchNorm_39_out->get_data_handle();
        float *pConv2D_119 = (float*)Conv2D_119_out->get_data_handle();
        float *pConv2D_120 = (float*)Conv2D_120_out->get_data_handle();
        float *pConv2D_121 = (float*)Conv2D_121_out->get_data_handle();
        float *pAdd_39 = (float*)Add_39_out->get_data_handle();
        float *pBatchNorm_40 = (float*)BatchNorm_40_out->get_data_handle();
        float *pConv2D_122 = (float*)Conv2D_122_out->get_data_handle();
        float *pConv2D_123 = (float*)Conv2D_123_out->get_data_handle();
        float *pConv2D_124 = (float*)Conv2D_124_out->get_data_handle();
        float *pAdd_40 = (float*)Add_40_out->get_data_handle();
        float *pBatchNorm_41 = (float*)BatchNorm_41_out->get_data_handle();
        float *pConv2D_125 = (float*)Conv2D_125_out->get_data_handle();
        float *pConv2D_126 = (float*)Conv2D_126_out->get_data_handle();
        float *pConv2D_127 = (float*)Conv2D_127_out->get_data_handle();
        float *pAdd_41 = (float*)Add_41_out->get_data_handle();
        float *pBatchNorm_42 = (float*)BatchNorm_42_out->get_data_handle();
        float *pConv2D_128 = (float*)Conv2D_128_out->get_data_handle();
        float *pConv2D_129 = (float*)Conv2D_129_out->get_data_handle();
        float *pConv2D_130 = (float*)Conv2D_130_out->get_data_handle();
        float *pConv2D_131 = (float*)Conv2D_131_out->get_data_handle();
        float *pAdd_42 = (float*)Add_42_out->get_data_handle();
        float *pBatchNorm_43 = (float*)BatchNorm_43_out->get_data_handle();
        float *pConv2D_132 = (float*)Conv2D_132_out->get_data_handle();
        float *pConv2D_133 = (float*)Conv2D_133_out->get_data_handle();
        float *pConv2D_134 = (float*)Conv2D_134_out->get_data_handle();
        float *pAdd_43 = (float*)Add_43_out->get_data_handle();
        float *pBatchNorm_44 = (float*)BatchNorm_44_out->get_data_handle();
        float *pAvgPool_1 = (float*)AvgPool_1_out->get_data_handle();
        float *pConv2D_135 = (float*)Conv2D_135_out->get_data_handle();

        printf("---------------result---------------\n\n");
        printf("    --< layer1/layer1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_1[0], pConv2D_1[1], pConv2D_1[2], pConv2D_1[3], pConv2D_1[4], pConv2D_1[5], pConv2D_1[6], pConv2D_1[7], pConv2D_1[8], pConv2D_1[9]);
        printf("    --< layer2/MaxPool2D/MaxPool >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pMaxPool_1[0], pMaxPool_1[1], pMaxPool_1[2], pMaxPool_1[3], pMaxPool_1[4], pMaxPool_1[5], pMaxPool_1[6], pMaxPool_1[7], pMaxPool_1[8], pMaxPool_1[9]);
        printf("    --< layer2/block0/common_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_1[0], pBatchNorm_1[1], pBatchNorm_1[2], pBatchNorm_1[3], pBatchNorm_1[4], pBatchNorm_1[5], pBatchNorm_1[6], pBatchNorm_1[7], pBatchNorm_1[8], pBatchNorm_1[9]);
        printf("    --< layer2/block0/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_2[0], pConv2D_2[1], pConv2D_2[2], pConv2D_2[3], pConv2D_2[4], pConv2D_2[5], pConv2D_2[6], pConv2D_2[7], pConv2D_2[8], pConv2D_2[9]);
        printf("    --< layer2/block0/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_3[0], pConv2D_3[1], pConv2D_3[2], pConv2D_3[3], pConv2D_3[4], pConv2D_3[5], pConv2D_3[6], pConv2D_3[7], pConv2D_3[8], pConv2D_3[9]);
        printf("    --< layer2/block0/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_4[0], pConv2D_4[1], pConv2D_4[2], pConv2D_4[3], pConv2D_4[4], pConv2D_4[5], pConv2D_4[6], pConv2D_4[7], pConv2D_4[8], pConv2D_4[9]);
        printf("    --< layer2/block0/shortcut/sub_sc/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_5[0], pConv2D_5[1], pConv2D_5[2], pConv2D_5[3], pConv2D_5[4], pConv2D_5[5], pConv2D_5[6], pConv2D_5[7], pConv2D_5[8], pConv2D_5[9]);
        printf("    --< layer2/block0/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_1[0], pAdd_1[1], pAdd_1[2], pAdd_1[3], pAdd_1[4], pAdd_1[5], pAdd_1[6], pAdd_1[7], pAdd_1[8], pAdd_1[9]);
        printf("    --< layer2/block1/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_2[0], pBatchNorm_2[1], pBatchNorm_2[2], pBatchNorm_2[3], pBatchNorm_2[4], pBatchNorm_2[5], pBatchNorm_2[6], pBatchNorm_2[7], pBatchNorm_2[8], pBatchNorm_2[9]);
        printf("    --< layer2/block1/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_6[0], pConv2D_6[1], pConv2D_6[2], pConv2D_6[3], pConv2D_6[4], pConv2D_6[5], pConv2D_6[6], pConv2D_6[7], pConv2D_6[8], pConv2D_6[9]);
        printf("    --< layer2/block1/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_7[0], pConv2D_7[1], pConv2D_7[2], pConv2D_7[3], pConv2D_7[4], pConv2D_7[5], pConv2D_7[6], pConv2D_7[7], pConv2D_7[8], pConv2D_7[9]);
        printf("    --< layer2/block1/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_8[0], pConv2D_8[1], pConv2D_8[2], pConv2D_8[3], pConv2D_8[4], pConv2D_8[5], pConv2D_8[6], pConv2D_8[7], pConv2D_8[8], pConv2D_8[9]);
        printf("    --< layer2/block1/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_2[0], pAdd_2[1], pAdd_2[2], pAdd_2[3], pAdd_2[4], pAdd_2[5], pAdd_2[6], pAdd_2[7], pAdd_2[8], pAdd_2[9]);
        printf("    --< layer3/block0/common_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_3[0], pBatchNorm_3[1], pBatchNorm_3[2], pBatchNorm_3[3], pBatchNorm_3[4], pBatchNorm_3[5], pBatchNorm_3[6], pBatchNorm_3[7], pBatchNorm_3[8], pBatchNorm_3[9]);
        printf("    --< layer3/block0/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_9[0], pConv2D_9[1], pConv2D_9[2], pConv2D_9[3], pConv2D_9[4], pConv2D_9[5], pConv2D_9[6], pConv2D_9[7], pConv2D_9[8], pConv2D_9[9]);
        printf("    --< layer3/block0/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_10[0], pConv2D_10[1], pConv2D_10[2], pConv2D_10[3], pConv2D_10[4], pConv2D_10[5], pConv2D_10[6], pConv2D_10[7], pConv2D_10[8], pConv2D_10[9]);
        printf("    --< layer3/block0/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_11[0], pConv2D_11[1], pConv2D_11[2], pConv2D_11[3], pConv2D_11[4], pConv2D_11[5], pConv2D_11[6], pConv2D_11[7], pConv2D_11[8], pConv2D_11[9]);
        printf("    --< layer3/block0/shortcut/sub_sc/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_12[0], pConv2D_12[1], pConv2D_12[2], pConv2D_12[3], pConv2D_12[4], pConv2D_12[5], pConv2D_12[6], pConv2D_12[7], pConv2D_12[8], pConv2D_12[9]);
        printf("    --< layer3/block0/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_3[0], pAdd_3[1], pAdd_3[2], pAdd_3[3], pAdd_3[4], pAdd_3[5], pAdd_3[6], pAdd_3[7], pAdd_3[8], pAdd_3[9]);
        printf("    --< layer3/block1/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_4[0], pBatchNorm_4[1], pBatchNorm_4[2], pBatchNorm_4[3], pBatchNorm_4[4], pBatchNorm_4[5], pBatchNorm_4[6], pBatchNorm_4[7], pBatchNorm_4[8], pBatchNorm_4[9]);
        printf("    --< layer3/block1/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_13[0], pConv2D_13[1], pConv2D_13[2], pConv2D_13[3], pConv2D_13[4], pConv2D_13[5], pConv2D_13[6], pConv2D_13[7], pConv2D_13[8], pConv2D_13[9]);
        printf("    --< layer3/block1/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_14[0], pConv2D_14[1], pConv2D_14[2], pConv2D_14[3], pConv2D_14[4], pConv2D_14[5], pConv2D_14[6], pConv2D_14[7], pConv2D_14[8], pConv2D_14[9]);
        printf("    --< layer3/block1/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_15[0], pConv2D_15[1], pConv2D_15[2], pConv2D_15[3], pConv2D_15[4], pConv2D_15[5], pConv2D_15[6], pConv2D_15[7], pConv2D_15[8], pConv2D_15[9]);
        printf("    --< layer3/block1/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_4[0], pAdd_4[1], pAdd_4[2], pAdd_4[3], pAdd_4[4], pAdd_4[5], pAdd_4[6], pAdd_4[7], pAdd_4[8], pAdd_4[9]);
        printf("    --< layer3/block2/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_5[0], pBatchNorm_5[1], pBatchNorm_5[2], pBatchNorm_5[3], pBatchNorm_5[4], pBatchNorm_5[5], pBatchNorm_5[6], pBatchNorm_5[7], pBatchNorm_5[8], pBatchNorm_5[9]);
        printf("    --< layer3/block2/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_16[0], pConv2D_16[1], pConv2D_16[2], pConv2D_16[3], pConv2D_16[4], pConv2D_16[5], pConv2D_16[6], pConv2D_16[7], pConv2D_16[8], pConv2D_16[9]);
        printf("    --< layer3/block2/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_17[0], pConv2D_17[1], pConv2D_17[2], pConv2D_17[3], pConv2D_17[4], pConv2D_17[5], pConv2D_17[6], pConv2D_17[7], pConv2D_17[8], pConv2D_17[9]);
        printf("    --< layer3/block2/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_18[0], pConv2D_18[1], pConv2D_18[2], pConv2D_18[3], pConv2D_18[4], pConv2D_18[5], pConv2D_18[6], pConv2D_18[7], pConv2D_18[8], pConv2D_18[9]);
        printf("    --< layer3/block2/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_5[0], pAdd_5[1], pAdd_5[2], pAdd_5[3], pAdd_5[4], pAdd_5[5], pAdd_5[6], pAdd_5[7], pAdd_5[8], pAdd_5[9]);
        printf("    --< layer3/block3/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_6[0], pBatchNorm_6[1], pBatchNorm_6[2], pBatchNorm_6[3], pBatchNorm_6[4], pBatchNorm_6[5], pBatchNorm_6[6], pBatchNorm_6[7], pBatchNorm_6[8], pBatchNorm_6[9]);
        printf("    --< layer3/block3/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_19[0], pConv2D_19[1], pConv2D_19[2], pConv2D_19[3], pConv2D_19[4], pConv2D_19[5], pConv2D_19[6], pConv2D_19[7], pConv2D_19[8], pConv2D_19[9]);
        printf("    --< layer3/block3/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_20[0], pConv2D_20[1], pConv2D_20[2], pConv2D_20[3], pConv2D_20[4], pConv2D_20[5], pConv2D_20[6], pConv2D_20[7], pConv2D_20[8], pConv2D_20[9]);
        printf("    --< layer3/block3/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_21[0], pConv2D_21[1], pConv2D_21[2], pConv2D_21[3], pConv2D_21[4], pConv2D_21[5], pConv2D_21[6], pConv2D_21[7], pConv2D_21[8], pConv2D_21[9]);
        printf("    --< layer3/block3/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_6[0], pAdd_6[1], pAdd_6[2], pAdd_6[3], pAdd_6[4], pAdd_6[5], pAdd_6[6], pAdd_6[7], pAdd_6[8], pAdd_6[9]);
        printf("    --< layer3/block4/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_7[0], pBatchNorm_7[1], pBatchNorm_7[2], pBatchNorm_7[3], pBatchNorm_7[4], pBatchNorm_7[5], pBatchNorm_7[6], pBatchNorm_7[7], pBatchNorm_7[8], pBatchNorm_7[9]);
        printf("    --< layer3/block4/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_22[0], pConv2D_22[1], pConv2D_22[2], pConv2D_22[3], pConv2D_22[4], pConv2D_22[5], pConv2D_22[6], pConv2D_22[7], pConv2D_22[8], pConv2D_22[9]);
        printf("    --< layer3/block4/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_23[0], pConv2D_23[1], pConv2D_23[2], pConv2D_23[3], pConv2D_23[4], pConv2D_23[5], pConv2D_23[6], pConv2D_23[7], pConv2D_23[8], pConv2D_23[9]);
        printf("    --< layer3/block4/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_24[0], pConv2D_24[1], pConv2D_24[2], pConv2D_24[3], pConv2D_24[4], pConv2D_24[5], pConv2D_24[6], pConv2D_24[7], pConv2D_24[8], pConv2D_24[9]);
        printf("    --< layer3/block4/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_7[0], pAdd_7[1], pAdd_7[2], pAdd_7[3], pAdd_7[4], pAdd_7[5], pAdd_7[6], pAdd_7[7], pAdd_7[8], pAdd_7[9]);
        printf("    --< layer3/block5/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_8[0], pBatchNorm_8[1], pBatchNorm_8[2], pBatchNorm_8[3], pBatchNorm_8[4], pBatchNorm_8[5], pBatchNorm_8[6], pBatchNorm_8[7], pBatchNorm_8[8], pBatchNorm_8[9]);
        printf("    --< layer3/block5/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_25[0], pConv2D_25[1], pConv2D_25[2], pConv2D_25[3], pConv2D_25[4], pConv2D_25[5], pConv2D_25[6], pConv2D_25[7], pConv2D_25[8], pConv2D_25[9]);
        printf("    --< layer3/block5/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_26[0], pConv2D_26[1], pConv2D_26[2], pConv2D_26[3], pConv2D_26[4], pConv2D_26[5], pConv2D_26[6], pConv2D_26[7], pConv2D_26[8], pConv2D_26[9]);
        printf("    --< layer3/block5/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_27[0], pConv2D_27[1], pConv2D_27[2], pConv2D_27[3], pConv2D_27[4], pConv2D_27[5], pConv2D_27[6], pConv2D_27[7], pConv2D_27[8], pConv2D_27[9]);
        printf("    --< layer3/block5/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_8[0], pAdd_8[1], pAdd_8[2], pAdd_8[3], pAdd_8[4], pAdd_8[5], pAdd_8[6], pAdd_8[7], pAdd_8[8], pAdd_8[9]);
        printf("    --< layer3/block6/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_9[0], pBatchNorm_9[1], pBatchNorm_9[2], pBatchNorm_9[3], pBatchNorm_9[4], pBatchNorm_9[5], pBatchNorm_9[6], pBatchNorm_9[7], pBatchNorm_9[8], pBatchNorm_9[9]);
        printf("    --< layer3/block6/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_28[0], pConv2D_28[1], pConv2D_28[2], pConv2D_28[3], pConv2D_28[4], pConv2D_28[5], pConv2D_28[6], pConv2D_28[7], pConv2D_28[8], pConv2D_28[9]);
        printf("    --< layer3/block6/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_29[0], pConv2D_29[1], pConv2D_29[2], pConv2D_29[3], pConv2D_29[4], pConv2D_29[5], pConv2D_29[6], pConv2D_29[7], pConv2D_29[8], pConv2D_29[9]);
        printf("    --< layer3/block6/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_30[0], pConv2D_30[1], pConv2D_30[2], pConv2D_30[3], pConv2D_30[4], pConv2D_30[5], pConv2D_30[6], pConv2D_30[7], pConv2D_30[8], pConv2D_30[9]);
        printf("    --< layer3/block6/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_9[0], pAdd_9[1], pAdd_9[2], pAdd_9[3], pAdd_9[4], pAdd_9[5], pAdd_9[6], pAdd_9[7], pAdd_9[8], pAdd_9[9]);
        printf("    --< layer4/block0/common_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_10[0], pBatchNorm_10[1], pBatchNorm_10[2], pBatchNorm_10[3], pBatchNorm_10[4], pBatchNorm_10[5], pBatchNorm_10[6], pBatchNorm_10[7], pBatchNorm_10[8], pBatchNorm_10[9]);
        printf("    --< layer4/block0/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_31[0], pConv2D_31[1], pConv2D_31[2], pConv2D_31[3], pConv2D_31[4], pConv2D_31[5], pConv2D_31[6], pConv2D_31[7], pConv2D_31[8], pConv2D_31[9]);
        printf("    --< layer4/block0/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_32[0], pConv2D_32[1], pConv2D_32[2], pConv2D_32[3], pConv2D_32[4], pConv2D_32[5], pConv2D_32[6], pConv2D_32[7], pConv2D_32[8], pConv2D_32[9]);
        printf("    --< layer4/block0/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_33[0], pConv2D_33[1], pConv2D_33[2], pConv2D_33[3], pConv2D_33[4], pConv2D_33[5], pConv2D_33[6], pConv2D_33[7], pConv2D_33[8], pConv2D_33[9]);
        printf("    --< layer4/block0/shortcut/sub_sc/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_34[0], pConv2D_34[1], pConv2D_34[2], pConv2D_34[3], pConv2D_34[4], pConv2D_34[5], pConv2D_34[6], pConv2D_34[7], pConv2D_34[8], pConv2D_34[9]);
        printf("    --< layer4/block0/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_10[0], pAdd_10[1], pAdd_10[2], pAdd_10[3], pAdd_10[4], pAdd_10[5], pAdd_10[6], pAdd_10[7], pAdd_10[8], pAdd_10[9]);
        printf("    --< layer4/block1/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_11[0], pBatchNorm_11[1], pBatchNorm_11[2], pBatchNorm_11[3], pBatchNorm_11[4], pBatchNorm_11[5], pBatchNorm_11[6], pBatchNorm_11[7], pBatchNorm_11[8], pBatchNorm_11[9]);
        printf("    --< layer4/block1/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_35[0], pConv2D_35[1], pConv2D_35[2], pConv2D_35[3], pConv2D_35[4], pConv2D_35[5], pConv2D_35[6], pConv2D_35[7], pConv2D_35[8], pConv2D_35[9]);
        printf("    --< layer4/block1/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_36[0], pConv2D_36[1], pConv2D_36[2], pConv2D_36[3], pConv2D_36[4], pConv2D_36[5], pConv2D_36[6], pConv2D_36[7], pConv2D_36[8], pConv2D_36[9]);
        printf("    --< layer4/block1/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_37[0], pConv2D_37[1], pConv2D_37[2], pConv2D_37[3], pConv2D_37[4], pConv2D_37[5], pConv2D_37[6], pConv2D_37[7], pConv2D_37[8], pConv2D_37[9]);
        printf("    --< layer4/block1/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_11[0], pAdd_11[1], pAdd_11[2], pAdd_11[3], pAdd_11[4], pAdd_11[5], pAdd_11[6], pAdd_11[7], pAdd_11[8], pAdd_11[9]);
        printf("    --< layer4/block2/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_12[0], pBatchNorm_12[1], pBatchNorm_12[2], pBatchNorm_12[3], pBatchNorm_12[4], pBatchNorm_12[5], pBatchNorm_12[6], pBatchNorm_12[7], pBatchNorm_12[8], pBatchNorm_12[9]);
        printf("    --< layer4/block2/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_38[0], pConv2D_38[1], pConv2D_38[2], pConv2D_38[3], pConv2D_38[4], pConv2D_38[5], pConv2D_38[6], pConv2D_38[7], pConv2D_38[8], pConv2D_38[9]);
        printf("    --< layer4/block2/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_39[0], pConv2D_39[1], pConv2D_39[2], pConv2D_39[3], pConv2D_39[4], pConv2D_39[5], pConv2D_39[6], pConv2D_39[7], pConv2D_39[8], pConv2D_39[9]);
        printf("    --< layer4/block2/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_40[0], pConv2D_40[1], pConv2D_40[2], pConv2D_40[3], pConv2D_40[4], pConv2D_40[5], pConv2D_40[6], pConv2D_40[7], pConv2D_40[8], pConv2D_40[9]);
        printf("    --< layer4/block2/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_12[0], pAdd_12[1], pAdd_12[2], pAdd_12[3], pAdd_12[4], pAdd_12[5], pAdd_12[6], pAdd_12[7], pAdd_12[8], pAdd_12[9]);
        printf("    --< layer4/block3/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_13[0], pBatchNorm_13[1], pBatchNorm_13[2], pBatchNorm_13[3], pBatchNorm_13[4], pBatchNorm_13[5], pBatchNorm_13[6], pBatchNorm_13[7], pBatchNorm_13[8], pBatchNorm_13[9]);
        printf("    --< layer4/block3/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_41[0], pConv2D_41[1], pConv2D_41[2], pConv2D_41[3], pConv2D_41[4], pConv2D_41[5], pConv2D_41[6], pConv2D_41[7], pConv2D_41[8], pConv2D_41[9]);
        printf("    --< layer4/block3/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_42[0], pConv2D_42[1], pConv2D_42[2], pConv2D_42[3], pConv2D_42[4], pConv2D_42[5], pConv2D_42[6], pConv2D_42[7], pConv2D_42[8], pConv2D_42[9]);
        printf("    --< layer4/block3/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_43[0], pConv2D_43[1], pConv2D_43[2], pConv2D_43[3], pConv2D_43[4], pConv2D_43[5], pConv2D_43[6], pConv2D_43[7], pConv2D_43[8], pConv2D_43[9]);
        printf("    --< layer4/block3/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_13[0], pAdd_13[1], pAdd_13[2], pAdd_13[3], pAdd_13[4], pAdd_13[5], pAdd_13[6], pAdd_13[7], pAdd_13[8], pAdd_13[9]);
        printf("    --< layer4/block4/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_14[0], pBatchNorm_14[1], pBatchNorm_14[2], pBatchNorm_14[3], pBatchNorm_14[4], pBatchNorm_14[5], pBatchNorm_14[6], pBatchNorm_14[7], pBatchNorm_14[8], pBatchNorm_14[9]);
        printf("    --< layer4/block4/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_44[0], pConv2D_44[1], pConv2D_44[2], pConv2D_44[3], pConv2D_44[4], pConv2D_44[5], pConv2D_44[6], pConv2D_44[7], pConv2D_44[8], pConv2D_44[9]);
        printf("    --< layer4/block4/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_45[0], pConv2D_45[1], pConv2D_45[2], pConv2D_45[3], pConv2D_45[4], pConv2D_45[5], pConv2D_45[6], pConv2D_45[7], pConv2D_45[8], pConv2D_45[9]);
        printf("    --< layer4/block4/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_46[0], pConv2D_46[1], pConv2D_46[2], pConv2D_46[3], pConv2D_46[4], pConv2D_46[5], pConv2D_46[6], pConv2D_46[7], pConv2D_46[8], pConv2D_46[9]);
        printf("    --< layer4/block4/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_14[0], pAdd_14[1], pAdd_14[2], pAdd_14[3], pAdd_14[4], pAdd_14[5], pAdd_14[6], pAdd_14[7], pAdd_14[8], pAdd_14[9]);
        printf("    --< layer4/block5/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_15[0], pBatchNorm_15[1], pBatchNorm_15[2], pBatchNorm_15[3], pBatchNorm_15[4], pBatchNorm_15[5], pBatchNorm_15[6], pBatchNorm_15[7], pBatchNorm_15[8], pBatchNorm_15[9]);
        printf("    --< layer4/block5/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_47[0], pConv2D_47[1], pConv2D_47[2], pConv2D_47[3], pConv2D_47[4], pConv2D_47[5], pConv2D_47[6], pConv2D_47[7], pConv2D_47[8], pConv2D_47[9]);
        printf("    --< layer4/block5/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_48[0], pConv2D_48[1], pConv2D_48[2], pConv2D_48[3], pConv2D_48[4], pConv2D_48[5], pConv2D_48[6], pConv2D_48[7], pConv2D_48[8], pConv2D_48[9]);
        printf("    --< layer4/block5/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_49[0], pConv2D_49[1], pConv2D_49[2], pConv2D_49[3], pConv2D_49[4], pConv2D_49[5], pConv2D_49[6], pConv2D_49[7], pConv2D_49[8], pConv2D_49[9]);
        printf("    --< layer4/block5/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_15[0], pAdd_15[1], pAdd_15[2], pAdd_15[3], pAdd_15[4], pAdd_15[5], pAdd_15[6], pAdd_15[7], pAdd_15[8], pAdd_15[9]);
        printf("    --< layer4/block6/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_16[0], pBatchNorm_16[1], pBatchNorm_16[2], pBatchNorm_16[3], pBatchNorm_16[4], pBatchNorm_16[5], pBatchNorm_16[6], pBatchNorm_16[7], pBatchNorm_16[8], pBatchNorm_16[9]);
        printf("    --< layer4/block6/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_50[0], pConv2D_50[1], pConv2D_50[2], pConv2D_50[3], pConv2D_50[4], pConv2D_50[5], pConv2D_50[6], pConv2D_50[7], pConv2D_50[8], pConv2D_50[9]);
        printf("    --< layer4/block6/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_51[0], pConv2D_51[1], pConv2D_51[2], pConv2D_51[3], pConv2D_51[4], pConv2D_51[5], pConv2D_51[6], pConv2D_51[7], pConv2D_51[8], pConv2D_51[9]);
        printf("    --< layer4/block6/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_52[0], pConv2D_52[1], pConv2D_52[2], pConv2D_52[3], pConv2D_52[4], pConv2D_52[5], pConv2D_52[6], pConv2D_52[7], pConv2D_52[8], pConv2D_52[9]);
        printf("    --< layer4/block6/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_16[0], pAdd_16[1], pAdd_16[2], pAdd_16[3], pAdd_16[4], pAdd_16[5], pAdd_16[6], pAdd_16[7], pAdd_16[8], pAdd_16[9]);
        printf("    --< layer4/block7/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_17[0], pBatchNorm_17[1], pBatchNorm_17[2], pBatchNorm_17[3], pBatchNorm_17[4], pBatchNorm_17[5], pBatchNorm_17[6], pBatchNorm_17[7], pBatchNorm_17[8], pBatchNorm_17[9]);
        printf("    --< layer4/block7/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_53[0], pConv2D_53[1], pConv2D_53[2], pConv2D_53[3], pConv2D_53[4], pConv2D_53[5], pConv2D_53[6], pConv2D_53[7], pConv2D_53[8], pConv2D_53[9]);
        printf("    --< layer4/block7/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_54[0], pConv2D_54[1], pConv2D_54[2], pConv2D_54[3], pConv2D_54[4], pConv2D_54[5], pConv2D_54[6], pConv2D_54[7], pConv2D_54[8], pConv2D_54[9]);
        printf("    --< layer4/block7/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_55[0], pConv2D_55[1], pConv2D_55[2], pConv2D_55[3], pConv2D_55[4], pConv2D_55[5], pConv2D_55[6], pConv2D_55[7], pConv2D_55[8], pConv2D_55[9]);
        printf("    --< layer4/block7/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_17[0], pAdd_17[1], pAdd_17[2], pAdd_17[3], pAdd_17[4], pAdd_17[5], pAdd_17[6], pAdd_17[7], pAdd_17[8], pAdd_17[9]);
        printf("    --< layer4/block8/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_18[0], pBatchNorm_18[1], pBatchNorm_18[2], pBatchNorm_18[3], pBatchNorm_18[4], pBatchNorm_18[5], pBatchNorm_18[6], pBatchNorm_18[7], pBatchNorm_18[8], pBatchNorm_18[9]);
        printf("    --< layer4/block8/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_56[0], pConv2D_56[1], pConv2D_56[2], pConv2D_56[3], pConv2D_56[4], pConv2D_56[5], pConv2D_56[6], pConv2D_56[7], pConv2D_56[8], pConv2D_56[9]);
        printf("    --< layer4/block8/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_57[0], pConv2D_57[1], pConv2D_57[2], pConv2D_57[3], pConv2D_57[4], pConv2D_57[5], pConv2D_57[6], pConv2D_57[7], pConv2D_57[8], pConv2D_57[9]);
        printf("    --< layer4/block8/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_58[0], pConv2D_58[1], pConv2D_58[2], pConv2D_58[3], pConv2D_58[4], pConv2D_58[5], pConv2D_58[6], pConv2D_58[7], pConv2D_58[8], pConv2D_58[9]);
        printf("    --< layer4/block8/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_18[0], pAdd_18[1], pAdd_18[2], pAdd_18[3], pAdd_18[4], pAdd_18[5], pAdd_18[6], pAdd_18[7], pAdd_18[8], pAdd_18[9]);
        printf("    --< layer4/block9/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_19[0], pBatchNorm_19[1], pBatchNorm_19[2], pBatchNorm_19[3], pBatchNorm_19[4], pBatchNorm_19[5], pBatchNorm_19[6], pBatchNorm_19[7], pBatchNorm_19[8], pBatchNorm_19[9]);
        printf("    --< layer4/block9/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_59[0], pConv2D_59[1], pConv2D_59[2], pConv2D_59[3], pConv2D_59[4], pConv2D_59[5], pConv2D_59[6], pConv2D_59[7], pConv2D_59[8], pConv2D_59[9]);
        printf("    --< layer4/block9/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_60[0], pConv2D_60[1], pConv2D_60[2], pConv2D_60[3], pConv2D_60[4], pConv2D_60[5], pConv2D_60[6], pConv2D_60[7], pConv2D_60[8], pConv2D_60[9]);
        printf("    --< layer4/block9/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_61[0], pConv2D_61[1], pConv2D_61[2], pConv2D_61[3], pConv2D_61[4], pConv2D_61[5], pConv2D_61[6], pConv2D_61[7], pConv2D_61[8], pConv2D_61[9]);
        printf("    --< layer4/block9/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_19[0], pAdd_19[1], pAdd_19[2], pAdd_19[3], pAdd_19[4], pAdd_19[5], pAdd_19[6], pAdd_19[7], pAdd_19[8], pAdd_19[9]);
        printf("    --< layer4/block10/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_20[0], pBatchNorm_20[1], pBatchNorm_20[2], pBatchNorm_20[3], pBatchNorm_20[4], pBatchNorm_20[5], pBatchNorm_20[6], pBatchNorm_20[7], pBatchNorm_20[8], pBatchNorm_20[9]);
        printf("    --< layer4/block10/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_62[0], pConv2D_62[1], pConv2D_62[2], pConv2D_62[3], pConv2D_62[4], pConv2D_62[5], pConv2D_62[6], pConv2D_62[7], pConv2D_62[8], pConv2D_62[9]);
        printf("    --< layer4/block10/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_63[0], pConv2D_63[1], pConv2D_63[2], pConv2D_63[3], pConv2D_63[4], pConv2D_63[5], pConv2D_63[6], pConv2D_63[7], pConv2D_63[8], pConv2D_63[9]);
        printf("    --< layer4/block10/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_64[0], pConv2D_64[1], pConv2D_64[2], pConv2D_64[3], pConv2D_64[4], pConv2D_64[5], pConv2D_64[6], pConv2D_64[7], pConv2D_64[8], pConv2D_64[9]);
        printf("    --< layer4/block10/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_20[0], pAdd_20[1], pAdd_20[2], pAdd_20[3], pAdd_20[4], pAdd_20[5], pAdd_20[6], pAdd_20[7], pAdd_20[8], pAdd_20[9]);
        printf("    --< layer4/block11/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_21[0], pBatchNorm_21[1], pBatchNorm_21[2], pBatchNorm_21[3], pBatchNorm_21[4], pBatchNorm_21[5], pBatchNorm_21[6], pBatchNorm_21[7], pBatchNorm_21[8], pBatchNorm_21[9]);
        printf("    --< layer4/block11/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_65[0], pConv2D_65[1], pConv2D_65[2], pConv2D_65[3], pConv2D_65[4], pConv2D_65[5], pConv2D_65[6], pConv2D_65[7], pConv2D_65[8], pConv2D_65[9]);
        printf("    --< layer4/block11/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_66[0], pConv2D_66[1], pConv2D_66[2], pConv2D_66[3], pConv2D_66[4], pConv2D_66[5], pConv2D_66[6], pConv2D_66[7], pConv2D_66[8], pConv2D_66[9]);
        printf("    --< layer4/block11/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_67[0], pConv2D_67[1], pConv2D_67[2], pConv2D_67[3], pConv2D_67[4], pConv2D_67[5], pConv2D_67[6], pConv2D_67[7], pConv2D_67[8], pConv2D_67[9]);
        printf("    --< layer4/block11/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_21[0], pAdd_21[1], pAdd_21[2], pAdd_21[3], pAdd_21[4], pAdd_21[5], pAdd_21[6], pAdd_21[7], pAdd_21[8], pAdd_21[9]);
        printf("    --< layer4/block12/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_22[0], pBatchNorm_22[1], pBatchNorm_22[2], pBatchNorm_22[3], pBatchNorm_22[4], pBatchNorm_22[5], pBatchNorm_22[6], pBatchNorm_22[7], pBatchNorm_22[8], pBatchNorm_22[9]);
        printf("    --< layer4/block12/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_68[0], pConv2D_68[1], pConv2D_68[2], pConv2D_68[3], pConv2D_68[4], pConv2D_68[5], pConv2D_68[6], pConv2D_68[7], pConv2D_68[8], pConv2D_68[9]);
        printf("    --< layer4/block12/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_69[0], pConv2D_69[1], pConv2D_69[2], pConv2D_69[3], pConv2D_69[4], pConv2D_69[5], pConv2D_69[6], pConv2D_69[7], pConv2D_69[8], pConv2D_69[9]);
        printf("    --< layer4/block12/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_70[0], pConv2D_70[1], pConv2D_70[2], pConv2D_70[3], pConv2D_70[4], pConv2D_70[5], pConv2D_70[6], pConv2D_70[7], pConv2D_70[8], pConv2D_70[9]);
        printf("    --< layer4/block12/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_22[0], pAdd_22[1], pAdd_22[2], pAdd_22[3], pAdd_22[4], pAdd_22[5], pAdd_22[6], pAdd_22[7], pAdd_22[8], pAdd_22[9]);
        printf("    --< layer4/block13/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_23[0], pBatchNorm_23[1], pBatchNorm_23[2], pBatchNorm_23[3], pBatchNorm_23[4], pBatchNorm_23[5], pBatchNorm_23[6], pBatchNorm_23[7], pBatchNorm_23[8], pBatchNorm_23[9]);
        printf("    --< layer4/block13/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_71[0], pConv2D_71[1], pConv2D_71[2], pConv2D_71[3], pConv2D_71[4], pConv2D_71[5], pConv2D_71[6], pConv2D_71[7], pConv2D_71[8], pConv2D_71[9]);
        printf("    --< layer4/block13/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_72[0], pConv2D_72[1], pConv2D_72[2], pConv2D_72[3], pConv2D_72[4], pConv2D_72[5], pConv2D_72[6], pConv2D_72[7], pConv2D_72[8], pConv2D_72[9]);
        printf("    --< layer4/block13/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_73[0], pConv2D_73[1], pConv2D_73[2], pConv2D_73[3], pConv2D_73[4], pConv2D_73[5], pConv2D_73[6], pConv2D_73[7], pConv2D_73[8], pConv2D_73[9]);
        printf("    --< layer4/block13/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_23[0], pAdd_23[1], pAdd_23[2], pAdd_23[3], pAdd_23[4], pAdd_23[5], pAdd_23[6], pAdd_23[7], pAdd_23[8], pAdd_23[9]);
        printf("    --< layer4/block14/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_24[0], pBatchNorm_24[1], pBatchNorm_24[2], pBatchNorm_24[3], pBatchNorm_24[4], pBatchNorm_24[5], pBatchNorm_24[6], pBatchNorm_24[7], pBatchNorm_24[8], pBatchNorm_24[9]);
        printf("    --< layer4/block14/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_74[0], pConv2D_74[1], pConv2D_74[2], pConv2D_74[3], pConv2D_74[4], pConv2D_74[5], pConv2D_74[6], pConv2D_74[7], pConv2D_74[8], pConv2D_74[9]);
        printf("    --< layer4/block14/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_75[0], pConv2D_75[1], pConv2D_75[2], pConv2D_75[3], pConv2D_75[4], pConv2D_75[5], pConv2D_75[6], pConv2D_75[7], pConv2D_75[8], pConv2D_75[9]);
        printf("    --< layer4/block14/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_76[0], pConv2D_76[1], pConv2D_76[2], pConv2D_76[3], pConv2D_76[4], pConv2D_76[5], pConv2D_76[6], pConv2D_76[7], pConv2D_76[8], pConv2D_76[9]);
        printf("    --< layer4/block14/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_24[0], pAdd_24[1], pAdd_24[2], pAdd_24[3], pAdd_24[4], pAdd_24[5], pAdd_24[6], pAdd_24[7], pAdd_24[8], pAdd_24[9]);
        printf("    --< layer4/block15/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_25[0], pBatchNorm_25[1], pBatchNorm_25[2], pBatchNorm_25[3], pBatchNorm_25[4], pBatchNorm_25[5], pBatchNorm_25[6], pBatchNorm_25[7], pBatchNorm_25[8], pBatchNorm_25[9]);
        printf("    --< layer4/block15/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_77[0], pConv2D_77[1], pConv2D_77[2], pConv2D_77[3], pConv2D_77[4], pConv2D_77[5], pConv2D_77[6], pConv2D_77[7], pConv2D_77[8], pConv2D_77[9]);
        printf("    --< layer4/block15/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_78[0], pConv2D_78[1], pConv2D_78[2], pConv2D_78[3], pConv2D_78[4], pConv2D_78[5], pConv2D_78[6], pConv2D_78[7], pConv2D_78[8], pConv2D_78[9]);
        printf("    --< layer4/block15/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_79[0], pConv2D_79[1], pConv2D_79[2], pConv2D_79[3], pConv2D_79[4], pConv2D_79[5], pConv2D_79[6], pConv2D_79[7], pConv2D_79[8], pConv2D_79[9]);
        printf("    --< layer4/block15/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_25[0], pAdd_25[1], pAdd_25[2], pAdd_25[3], pAdd_25[4], pAdd_25[5], pAdd_25[6], pAdd_25[7], pAdd_25[8], pAdd_25[9]);
        printf("    --< layer4/block16/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_26[0], pBatchNorm_26[1], pBatchNorm_26[2], pBatchNorm_26[3], pBatchNorm_26[4], pBatchNorm_26[5], pBatchNorm_26[6], pBatchNorm_26[7], pBatchNorm_26[8], pBatchNorm_26[9]);
        printf("    --< layer4/block16/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_80[0], pConv2D_80[1], pConv2D_80[2], pConv2D_80[3], pConv2D_80[4], pConv2D_80[5], pConv2D_80[6], pConv2D_80[7], pConv2D_80[8], pConv2D_80[9]);
        printf("    --< layer4/block16/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_81[0], pConv2D_81[1], pConv2D_81[2], pConv2D_81[3], pConv2D_81[4], pConv2D_81[5], pConv2D_81[6], pConv2D_81[7], pConv2D_81[8], pConv2D_81[9]);
        printf("    --< layer4/block16/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_82[0], pConv2D_82[1], pConv2D_82[2], pConv2D_82[3], pConv2D_82[4], pConv2D_82[5], pConv2D_82[6], pConv2D_82[7], pConv2D_82[8], pConv2D_82[9]);
        printf("    --< layer4/block16/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_26[0], pAdd_26[1], pAdd_26[2], pAdd_26[3], pAdd_26[4], pAdd_26[5], pAdd_26[6], pAdd_26[7], pAdd_26[8], pAdd_26[9]);
        printf("    --< layer4/block17/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_27[0], pBatchNorm_27[1], pBatchNorm_27[2], pBatchNorm_27[3], pBatchNorm_27[4], pBatchNorm_27[5], pBatchNorm_27[6], pBatchNorm_27[7], pBatchNorm_27[8], pBatchNorm_27[9]);
        printf("    --< layer4/block17/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_83[0], pConv2D_83[1], pConv2D_83[2], pConv2D_83[3], pConv2D_83[4], pConv2D_83[5], pConv2D_83[6], pConv2D_83[7], pConv2D_83[8], pConv2D_83[9]);
        printf("    --< layer4/block17/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_84[0], pConv2D_84[1], pConv2D_84[2], pConv2D_84[3], pConv2D_84[4], pConv2D_84[5], pConv2D_84[6], pConv2D_84[7], pConv2D_84[8], pConv2D_84[9]);
        printf("    --< layer4/block17/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_85[0], pConv2D_85[1], pConv2D_85[2], pConv2D_85[3], pConv2D_85[4], pConv2D_85[5], pConv2D_85[6], pConv2D_85[7], pConv2D_85[8], pConv2D_85[9]);
        printf("    --< layer4/block17/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_27[0], pAdd_27[1], pAdd_27[2], pAdd_27[3], pAdd_27[4], pAdd_27[5], pAdd_27[6], pAdd_27[7], pAdd_27[8], pAdd_27[9]);
        printf("    --< layer4/block18/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_28[0], pBatchNorm_28[1], pBatchNorm_28[2], pBatchNorm_28[3], pBatchNorm_28[4], pBatchNorm_28[5], pBatchNorm_28[6], pBatchNorm_28[7], pBatchNorm_28[8], pBatchNorm_28[9]);
        printf("    --< layer4/block18/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_86[0], pConv2D_86[1], pConv2D_86[2], pConv2D_86[3], pConv2D_86[4], pConv2D_86[5], pConv2D_86[6], pConv2D_86[7], pConv2D_86[8], pConv2D_86[9]);
        printf("    --< layer4/block18/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_87[0], pConv2D_87[1], pConv2D_87[2], pConv2D_87[3], pConv2D_87[4], pConv2D_87[5], pConv2D_87[6], pConv2D_87[7], pConv2D_87[8], pConv2D_87[9]);
        printf("    --< layer4/block18/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_88[0], pConv2D_88[1], pConv2D_88[2], pConv2D_88[3], pConv2D_88[4], pConv2D_88[5], pConv2D_88[6], pConv2D_88[7], pConv2D_88[8], pConv2D_88[9]);
        printf("    --< layer4/block18/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_28[0], pAdd_28[1], pAdd_28[2], pAdd_28[3], pAdd_28[4], pAdd_28[5], pAdd_28[6], pAdd_28[7], pAdd_28[8], pAdd_28[9]);
        printf("    --< layer4/block19/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_29[0], pBatchNorm_29[1], pBatchNorm_29[2], pBatchNorm_29[3], pBatchNorm_29[4], pBatchNorm_29[5], pBatchNorm_29[6], pBatchNorm_29[7], pBatchNorm_29[8], pBatchNorm_29[9]);
        printf("    --< layer4/block19/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_89[0], pConv2D_89[1], pConv2D_89[2], pConv2D_89[3], pConv2D_89[4], pConv2D_89[5], pConv2D_89[6], pConv2D_89[7], pConv2D_89[8], pConv2D_89[9]);
        printf("    --< layer4/block19/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_90[0], pConv2D_90[1], pConv2D_90[2], pConv2D_90[3], pConv2D_90[4], pConv2D_90[5], pConv2D_90[6], pConv2D_90[7], pConv2D_90[8], pConv2D_90[9]);
        printf("    --< layer4/block19/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_91[0], pConv2D_91[1], pConv2D_91[2], pConv2D_91[3], pConv2D_91[4], pConv2D_91[5], pConv2D_91[6], pConv2D_91[7], pConv2D_91[8], pConv2D_91[9]);
        printf("    --< layer4/block19/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_29[0], pAdd_29[1], pAdd_29[2], pAdd_29[3], pAdd_29[4], pAdd_29[5], pAdd_29[6], pAdd_29[7], pAdd_29[8], pAdd_29[9]);
        printf("    --< layer4/block20/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_30[0], pBatchNorm_30[1], pBatchNorm_30[2], pBatchNorm_30[3], pBatchNorm_30[4], pBatchNorm_30[5], pBatchNorm_30[6], pBatchNorm_30[7], pBatchNorm_30[8], pBatchNorm_30[9]);
        printf("    --< layer4/block20/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_92[0], pConv2D_92[1], pConv2D_92[2], pConv2D_92[3], pConv2D_92[4], pConv2D_92[5], pConv2D_92[6], pConv2D_92[7], pConv2D_92[8], pConv2D_92[9]);
        printf("    --< layer4/block20/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_93[0], pConv2D_93[1], pConv2D_93[2], pConv2D_93[3], pConv2D_93[4], pConv2D_93[5], pConv2D_93[6], pConv2D_93[7], pConv2D_93[8], pConv2D_93[9]);
        printf("    --< layer4/block20/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_94[0], pConv2D_94[1], pConv2D_94[2], pConv2D_94[3], pConv2D_94[4], pConv2D_94[5], pConv2D_94[6], pConv2D_94[7], pConv2D_94[8], pConv2D_94[9]);
        printf("    --< layer4/block20/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_30[0], pAdd_30[1], pAdd_30[2], pAdd_30[3], pAdd_30[4], pAdd_30[5], pAdd_30[6], pAdd_30[7], pAdd_30[8], pAdd_30[9]);
        printf("    --< layer4/block21/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_31[0], pBatchNorm_31[1], pBatchNorm_31[2], pBatchNorm_31[3], pBatchNorm_31[4], pBatchNorm_31[5], pBatchNorm_31[6], pBatchNorm_31[7], pBatchNorm_31[8], pBatchNorm_31[9]);
        printf("    --< layer4/block21/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_95[0], pConv2D_95[1], pConv2D_95[2], pConv2D_95[3], pConv2D_95[4], pConv2D_95[5], pConv2D_95[6], pConv2D_95[7], pConv2D_95[8], pConv2D_95[9]);
        printf("    --< layer4/block21/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_96[0], pConv2D_96[1], pConv2D_96[2], pConv2D_96[3], pConv2D_96[4], pConv2D_96[5], pConv2D_96[6], pConv2D_96[7], pConv2D_96[8], pConv2D_96[9]);
        printf("    --< layer4/block21/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_97[0], pConv2D_97[1], pConv2D_97[2], pConv2D_97[3], pConv2D_97[4], pConv2D_97[5], pConv2D_97[6], pConv2D_97[7], pConv2D_97[8], pConv2D_97[9]);
        printf("    --< layer4/block21/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_31[0], pAdd_31[1], pAdd_31[2], pAdd_31[3], pAdd_31[4], pAdd_31[5], pAdd_31[6], pAdd_31[7], pAdd_31[8], pAdd_31[9]);
        printf("    --< layer4/block22/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_32[0], pBatchNorm_32[1], pBatchNorm_32[2], pBatchNorm_32[3], pBatchNorm_32[4], pBatchNorm_32[5], pBatchNorm_32[6], pBatchNorm_32[7], pBatchNorm_32[8], pBatchNorm_32[9]);
        printf("    --< layer4/block22/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_98[0], pConv2D_98[1], pConv2D_98[2], pConv2D_98[3], pConv2D_98[4], pConv2D_98[5], pConv2D_98[6], pConv2D_98[7], pConv2D_98[8], pConv2D_98[9]);
        printf("    --< layer4/block22/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_99[0], pConv2D_99[1], pConv2D_99[2], pConv2D_99[3], pConv2D_99[4], pConv2D_99[5], pConv2D_99[6], pConv2D_99[7], pConv2D_99[8], pConv2D_99[9]);
        printf("    --< layer4/block22/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_100[0], pConv2D_100[1], pConv2D_100[2], pConv2D_100[3], pConv2D_100[4], pConv2D_100[5], pConv2D_100[6], pConv2D_100[7], pConv2D_100[8], pConv2D_100[9]);
        printf("    --< layer4/block22/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_32[0], pAdd_32[1], pAdd_32[2], pAdd_32[3], pAdd_32[4], pAdd_32[5], pAdd_32[6], pAdd_32[7], pAdd_32[8], pAdd_32[9]);
        printf("    --< layer4/block23/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_33[0], pBatchNorm_33[1], pBatchNorm_33[2], pBatchNorm_33[3], pBatchNorm_33[4], pBatchNorm_33[5], pBatchNorm_33[6], pBatchNorm_33[7], pBatchNorm_33[8], pBatchNorm_33[9]);
        printf("    --< layer4/block23/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_101[0], pConv2D_101[1], pConv2D_101[2], pConv2D_101[3], pConv2D_101[4], pConv2D_101[5], pConv2D_101[6], pConv2D_101[7], pConv2D_101[8], pConv2D_101[9]);
        printf("    --< layer4/block23/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_102[0], pConv2D_102[1], pConv2D_102[2], pConv2D_102[3], pConv2D_102[4], pConv2D_102[5], pConv2D_102[6], pConv2D_102[7], pConv2D_102[8], pConv2D_102[9]);
        printf("    --< layer4/block23/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_103[0], pConv2D_103[1], pConv2D_103[2], pConv2D_103[3], pConv2D_103[4], pConv2D_103[5], pConv2D_103[6], pConv2D_103[7], pConv2D_103[8], pConv2D_103[9]);
        printf("    --< layer4/block23/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_33[0], pAdd_33[1], pAdd_33[2], pAdd_33[3], pAdd_33[4], pAdd_33[5], pAdd_33[6], pAdd_33[7], pAdd_33[8], pAdd_33[9]);
        printf("    --< layer4/block24/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_34[0], pBatchNorm_34[1], pBatchNorm_34[2], pBatchNorm_34[3], pBatchNorm_34[4], pBatchNorm_34[5], pBatchNorm_34[6], pBatchNorm_34[7], pBatchNorm_34[8], pBatchNorm_34[9]);
        printf("    --< layer4/block24/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_104[0], pConv2D_104[1], pConv2D_104[2], pConv2D_104[3], pConv2D_104[4], pConv2D_104[5], pConv2D_104[6], pConv2D_104[7], pConv2D_104[8], pConv2D_104[9]);
        printf("    --< layer4/block24/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_105[0], pConv2D_105[1], pConv2D_105[2], pConv2D_105[3], pConv2D_105[4], pConv2D_105[5], pConv2D_105[6], pConv2D_105[7], pConv2D_105[8], pConv2D_105[9]);
        printf("    --< layer4/block24/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_106[0], pConv2D_106[1], pConv2D_106[2], pConv2D_106[3], pConv2D_106[4], pConv2D_106[5], pConv2D_106[6], pConv2D_106[7], pConv2D_106[8], pConv2D_106[9]);
        printf("    --< layer4/block24/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_34[0], pAdd_34[1], pAdd_34[2], pAdd_34[3], pAdd_34[4], pAdd_34[5], pAdd_34[6], pAdd_34[7], pAdd_34[8], pAdd_34[9]);
        printf("    --< layer4/block25/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_35[0], pBatchNorm_35[1], pBatchNorm_35[2], pBatchNorm_35[3], pBatchNorm_35[4], pBatchNorm_35[5], pBatchNorm_35[6], pBatchNorm_35[7], pBatchNorm_35[8], pBatchNorm_35[9]);
        printf("    --< layer4/block25/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_107[0], pConv2D_107[1], pConv2D_107[2], pConv2D_107[3], pConv2D_107[4], pConv2D_107[5], pConv2D_107[6], pConv2D_107[7], pConv2D_107[8], pConv2D_107[9]);
        printf("    --< layer4/block25/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_108[0], pConv2D_108[1], pConv2D_108[2], pConv2D_108[3], pConv2D_108[4], pConv2D_108[5], pConv2D_108[6], pConv2D_108[7], pConv2D_108[8], pConv2D_108[9]);
        printf("    --< layer4/block25/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_109[0], pConv2D_109[1], pConv2D_109[2], pConv2D_109[3], pConv2D_109[4], pConv2D_109[5], pConv2D_109[6], pConv2D_109[7], pConv2D_109[8], pConv2D_109[9]);
        printf("    --< layer4/block25/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_35[0], pAdd_35[1], pAdd_35[2], pAdd_35[3], pAdd_35[4], pAdd_35[5], pAdd_35[6], pAdd_35[7], pAdd_35[8], pAdd_35[9]);
        printf("    --< layer4/block26/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_36[0], pBatchNorm_36[1], pBatchNorm_36[2], pBatchNorm_36[3], pBatchNorm_36[4], pBatchNorm_36[5], pBatchNorm_36[6], pBatchNorm_36[7], pBatchNorm_36[8], pBatchNorm_36[9]);
        printf("    --< layer4/block26/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_110[0], pConv2D_110[1], pConv2D_110[2], pConv2D_110[3], pConv2D_110[4], pConv2D_110[5], pConv2D_110[6], pConv2D_110[7], pConv2D_110[8], pConv2D_110[9]);
        printf("    --< layer4/block26/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_111[0], pConv2D_111[1], pConv2D_111[2], pConv2D_111[3], pConv2D_111[4], pConv2D_111[5], pConv2D_111[6], pConv2D_111[7], pConv2D_111[8], pConv2D_111[9]);
        printf("    --< layer4/block26/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_112[0], pConv2D_112[1], pConv2D_112[2], pConv2D_112[3], pConv2D_112[4], pConv2D_112[5], pConv2D_112[6], pConv2D_112[7], pConv2D_112[8], pConv2D_112[9]);
        printf("    --< layer4/block26/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_36[0], pAdd_36[1], pAdd_36[2], pAdd_36[3], pAdd_36[4], pAdd_36[5], pAdd_36[6], pAdd_36[7], pAdd_36[8], pAdd_36[9]);
        printf("    --< layer4/block27/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_37[0], pBatchNorm_37[1], pBatchNorm_37[2], pBatchNorm_37[3], pBatchNorm_37[4], pBatchNorm_37[5], pBatchNorm_37[6], pBatchNorm_37[7], pBatchNorm_37[8], pBatchNorm_37[9]);
        printf("    --< layer4/block27/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_113[0], pConv2D_113[1], pConv2D_113[2], pConv2D_113[3], pConv2D_113[4], pConv2D_113[5], pConv2D_113[6], pConv2D_113[7], pConv2D_113[8], pConv2D_113[9]);
        printf("    --< layer4/block27/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_114[0], pConv2D_114[1], pConv2D_114[2], pConv2D_114[3], pConv2D_114[4], pConv2D_114[5], pConv2D_114[6], pConv2D_114[7], pConv2D_114[8], pConv2D_114[9]);
        printf("    --< layer4/block27/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_115[0], pConv2D_115[1], pConv2D_115[2], pConv2D_115[3], pConv2D_115[4], pConv2D_115[5], pConv2D_115[6], pConv2D_115[7], pConv2D_115[8], pConv2D_115[9]);
        printf("    --< layer4/block27/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_37[0], pAdd_37[1], pAdd_37[2], pAdd_37[3], pAdd_37[4], pAdd_37[5], pAdd_37[6], pAdd_37[7], pAdd_37[8], pAdd_37[9]);
        printf("    --< layer4/block28/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_38[0], pBatchNorm_38[1], pBatchNorm_38[2], pBatchNorm_38[3], pBatchNorm_38[4], pBatchNorm_38[5], pBatchNorm_38[6], pBatchNorm_38[7], pBatchNorm_38[8], pBatchNorm_38[9]);
        printf("    --< layer4/block28/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_116[0], pConv2D_116[1], pConv2D_116[2], pConv2D_116[3], pConv2D_116[4], pConv2D_116[5], pConv2D_116[6], pConv2D_116[7], pConv2D_116[8], pConv2D_116[9]);
        printf("    --< layer4/block28/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_117[0], pConv2D_117[1], pConv2D_117[2], pConv2D_117[3], pConv2D_117[4], pConv2D_117[5], pConv2D_117[6], pConv2D_117[7], pConv2D_117[8], pConv2D_117[9]);
        printf("    --< layer4/block28/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_118[0], pConv2D_118[1], pConv2D_118[2], pConv2D_118[3], pConv2D_118[4], pConv2D_118[5], pConv2D_118[6], pConv2D_118[7], pConv2D_118[8], pConv2D_118[9]);
        printf("    --< layer4/block28/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_38[0], pAdd_38[1], pAdd_38[2], pAdd_38[3], pAdd_38[4], pAdd_38[5], pAdd_38[6], pAdd_38[7], pAdd_38[8], pAdd_38[9]);
        printf("    --< layer4/block29/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_39[0], pBatchNorm_39[1], pBatchNorm_39[2], pBatchNorm_39[3], pBatchNorm_39[4], pBatchNorm_39[5], pBatchNorm_39[6], pBatchNorm_39[7], pBatchNorm_39[8], pBatchNorm_39[9]);
        printf("    --< layer4/block29/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_119[0], pConv2D_119[1], pConv2D_119[2], pConv2D_119[3], pConv2D_119[4], pConv2D_119[5], pConv2D_119[6], pConv2D_119[7], pConv2D_119[8], pConv2D_119[9]);
        printf("    --< layer4/block29/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_120[0], pConv2D_120[1], pConv2D_120[2], pConv2D_120[3], pConv2D_120[4], pConv2D_120[5], pConv2D_120[6], pConv2D_120[7], pConv2D_120[8], pConv2D_120[9]);
        printf("    --< layer4/block29/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_121[0], pConv2D_121[1], pConv2D_121[2], pConv2D_121[3], pConv2D_121[4], pConv2D_121[5], pConv2D_121[6], pConv2D_121[7], pConv2D_121[8], pConv2D_121[9]);
        printf("    --< layer4/block29/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_39[0], pAdd_39[1], pAdd_39[2], pAdd_39[3], pAdd_39[4], pAdd_39[5], pAdd_39[6], pAdd_39[7], pAdd_39[8], pAdd_39[9]);
        printf("    --< layer4/block30/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_40[0], pBatchNorm_40[1], pBatchNorm_40[2], pBatchNorm_40[3], pBatchNorm_40[4], pBatchNorm_40[5], pBatchNorm_40[6], pBatchNorm_40[7], pBatchNorm_40[8], pBatchNorm_40[9]);
        printf("    --< layer4/block30/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_122[0], pConv2D_122[1], pConv2D_122[2], pConv2D_122[3], pConv2D_122[4], pConv2D_122[5], pConv2D_122[6], pConv2D_122[7], pConv2D_122[8], pConv2D_122[9]);
        printf("    --< layer4/block30/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_123[0], pConv2D_123[1], pConv2D_123[2], pConv2D_123[3], pConv2D_123[4], pConv2D_123[5], pConv2D_123[6], pConv2D_123[7], pConv2D_123[8], pConv2D_123[9]);
        printf("    --< layer4/block30/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_124[0], pConv2D_124[1], pConv2D_124[2], pConv2D_124[3], pConv2D_124[4], pConv2D_124[5], pConv2D_124[6], pConv2D_124[7], pConv2D_124[8], pConv2D_124[9]);
        printf("    --< layer4/block30/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_40[0], pAdd_40[1], pAdd_40[2], pAdd_40[3], pAdd_40[4], pAdd_40[5], pAdd_40[6], pAdd_40[7], pAdd_40[8], pAdd_40[9]);
        printf("    --< layer4/block31/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_41[0], pBatchNorm_41[1], pBatchNorm_41[2], pBatchNorm_41[3], pBatchNorm_41[4], pBatchNorm_41[5], pBatchNorm_41[6], pBatchNorm_41[7], pBatchNorm_41[8], pBatchNorm_41[9]);
        printf("    --< layer4/block31/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_125[0], pConv2D_125[1], pConv2D_125[2], pConv2D_125[3], pConv2D_125[4], pConv2D_125[5], pConv2D_125[6], pConv2D_125[7], pConv2D_125[8], pConv2D_125[9]);
        printf("    --< layer4/block31/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_126[0], pConv2D_126[1], pConv2D_126[2], pConv2D_126[3], pConv2D_126[4], pConv2D_126[5], pConv2D_126[6], pConv2D_126[7], pConv2D_126[8], pConv2D_126[9]);
        printf("    --< layer4/block31/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_127[0], pConv2D_127[1], pConv2D_127[2], pConv2D_127[3], pConv2D_127[4], pConv2D_127[5], pConv2D_127[6], pConv2D_127[7], pConv2D_127[8], pConv2D_127[9]);
        printf("    --< layer4/block31/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_41[0], pAdd_41[1], pAdd_41[2], pAdd_41[3], pAdd_41[4], pAdd_41[5], pAdd_41[6], pAdd_41[7], pAdd_41[8], pAdd_41[9]);
        printf("    --< layer5/block0/common_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_42[0], pBatchNorm_42[1], pBatchNorm_42[2], pBatchNorm_42[3], pBatchNorm_42[4], pBatchNorm_42[5], pBatchNorm_42[6], pBatchNorm_42[7], pBatchNorm_42[8], pBatchNorm_42[9]);
        printf("    --< layer5/block0/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_128[0], pConv2D_128[1], pConv2D_128[2], pConv2D_128[3], pConv2D_128[4], pConv2D_128[5], pConv2D_128[6], pConv2D_128[7], pConv2D_128[8], pConv2D_128[9]);
        printf("    --< layer5/block0/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_129[0], pConv2D_129[1], pConv2D_129[2], pConv2D_129[3], pConv2D_129[4], pConv2D_129[5], pConv2D_129[6], pConv2D_129[7], pConv2D_129[8], pConv2D_129[9]);
        printf("    --< layer5/block0/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_130[0], pConv2D_130[1], pConv2D_130[2], pConv2D_130[3], pConv2D_130[4], pConv2D_130[5], pConv2D_130[6], pConv2D_130[7], pConv2D_130[8], pConv2D_130[9]);
        printf("    --< layer5/block0/shortcut/sub_sc/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_131[0], pConv2D_131[1], pConv2D_131[2], pConv2D_131[3], pConv2D_131[4], pConv2D_131[5], pConv2D_131[6], pConv2D_131[7], pConv2D_131[8], pConv2D_131[9]);
        printf("    --< layer5/block0/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_42[0], pAdd_42[1], pAdd_42[2], pAdd_42[3], pAdd_42[4], pAdd_42[5], pAdd_42[6], pAdd_42[7], pAdd_42[8], pAdd_42[9]);
        printf("    --< layer5/block1/residual_bn_relu/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_43[0], pBatchNorm_43[1], pBatchNorm_43[2], pBatchNorm_43[3], pBatchNorm_43[4], pBatchNorm_43[5], pBatchNorm_43[6], pBatchNorm_43[7], pBatchNorm_43[8], pBatchNorm_43[9]);
        printf("    --< layer5/block1/sub1/sub1_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_132[0], pConv2D_132[1], pConv2D_132[2], pConv2D_132[3], pConv2D_132[4], pConv2D_132[5], pConv2D_132[6], pConv2D_132[7], pConv2D_132[8], pConv2D_132[9]);
        printf("    --< layer5/block1/sub2/sub2_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_133[0], pConv2D_133[1], pConv2D_133[2], pConv2D_133[3], pConv2D_133[4], pConv2D_133[5], pConv2D_133[6], pConv2D_133[7], pConv2D_133[8], pConv2D_133[9]);
        printf("    --< layer5/block1/sub3/sub3_conv/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_134[0], pConv2D_134[1], pConv2D_134[2], pConv2D_134[3], pConv2D_134[4], pConv2D_134[5], pConv2D_134[6], pConv2D_134[7], pConv2D_134[8], pConv2D_134[9]);
        printf("    --< layer5/block1/shortcut/add >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAdd_43[0], pAdd_43[1], pAdd_43[2], pAdd_43[3], pAdd_43[4], pAdd_43[5], pAdd_43[6], pAdd_43[7], pAdd_43[8], pAdd_43[9]);
        printf("    --< avg_fc/fc_bn/FusedBatchNorm >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pBatchNorm_44[0], pBatchNorm_44[1], pBatchNorm_44[2], pBatchNorm_44[3], pBatchNorm_44[4], pBatchNorm_44[5], pBatchNorm_44[6], pBatchNorm_44[7], pBatchNorm_44[8], pBatchNorm_44[9]);
        printf("    --< avg_fc/AvgPool >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pAvgPool_1[0], pAvgPool_1[1], pAvgPool_1[2], pAvgPool_1[3], pAvgPool_1[4], pAvgPool_1[5], pAvgPool_1[6], pAvgPool_1[7], pAvgPool_1[8], pAvgPool_1[9]);
        printf("    --< avg_fc/fc6/Conv2D >--\n");
        printf("    >> [%f %f %f %f %f %f %f %f %f %f]\n\n", pConv2D_135[0], pConv2D_135[1], pConv2D_135[2], pConv2D_135[3], pConv2D_135[4], pConv2D_135[5], pConv2D_135[6], pConv2D_135[7], pConv2D_135[8], pConv2D_135[9]);
    }


    void release_io() {
        if(src_memory == NULL) return;

        delete src_memory; src_memory = NULL;
        delete Conv2D_1_out; Conv2D_1_out = NULL;
        delete MaxPool_1_out; MaxPool_1_out = NULL;
        delete BatchNorm_1_out; BatchNorm_1_out = NULL;
        delete Conv2D_2_out; Conv2D_2_out = NULL;
        delete Conv2D_3_out; Conv2D_3_out = NULL;
        delete Conv2D_4_out; Conv2D_4_out = NULL;
        delete Conv2D_5_out; Conv2D_5_out = NULL;
        delete Add_1_out; Add_1_out = NULL;
        delete BatchNorm_2_out; BatchNorm_2_out = NULL;
        delete Conv2D_6_out; Conv2D_6_out = NULL;
        delete Conv2D_7_out; Conv2D_7_out = NULL;
        delete Conv2D_8_out; Conv2D_8_out = NULL;
        delete Add_2_out; Add_2_out = NULL;
        delete BatchNorm_3_out; BatchNorm_3_out = NULL;
        delete Conv2D_9_out; Conv2D_9_out = NULL;
        delete Conv2D_10_out; Conv2D_10_out = NULL;
        delete Conv2D_11_out; Conv2D_11_out = NULL;
        delete Conv2D_12_out; Conv2D_12_out = NULL;
        delete Add_3_out; Add_3_out = NULL;
        delete BatchNorm_4_out; BatchNorm_4_out = NULL;
        delete Conv2D_13_out; Conv2D_13_out = NULL;
        delete Conv2D_14_out; Conv2D_14_out = NULL;
        delete Conv2D_15_out; Conv2D_15_out = NULL;
        delete Add_4_out; Add_4_out = NULL;
        delete BatchNorm_5_out; BatchNorm_5_out = NULL;
        delete Conv2D_16_out; Conv2D_16_out = NULL;
        delete Conv2D_17_out; Conv2D_17_out = NULL;
        delete Conv2D_18_out; Conv2D_18_out = NULL;
        delete Add_5_out; Add_5_out = NULL;
        delete BatchNorm_6_out; BatchNorm_6_out = NULL;
        delete Conv2D_19_out; Conv2D_19_out = NULL;
        delete Conv2D_20_out; Conv2D_20_out = NULL;
        delete Conv2D_21_out; Conv2D_21_out = NULL;
        delete Add_6_out; Add_6_out = NULL;
        delete BatchNorm_7_out; BatchNorm_7_out = NULL;
        delete Conv2D_22_out; Conv2D_22_out = NULL;
        delete Conv2D_23_out; Conv2D_23_out = NULL;
        delete Conv2D_24_out; Conv2D_24_out = NULL;
        delete Add_7_out; Add_7_out = NULL;
        delete BatchNorm_8_out; BatchNorm_8_out = NULL;
        delete Conv2D_25_out; Conv2D_25_out = NULL;
        delete Conv2D_26_out; Conv2D_26_out = NULL;
        delete Conv2D_27_out; Conv2D_27_out = NULL;
        delete Add_8_out; Add_8_out = NULL;
        delete BatchNorm_9_out; BatchNorm_9_out = NULL;
        delete Conv2D_28_out; Conv2D_28_out = NULL;
        delete Conv2D_29_out; Conv2D_29_out = NULL;
        delete Conv2D_30_out; Conv2D_30_out = NULL;
        delete Add_9_out; Add_9_out = NULL;
        delete BatchNorm_10_out; BatchNorm_10_out = NULL;
        delete Conv2D_31_out; Conv2D_31_out = NULL;
        delete Conv2D_32_out; Conv2D_32_out = NULL;
        delete Conv2D_33_out; Conv2D_33_out = NULL;
        delete Conv2D_34_out; Conv2D_34_out = NULL;
        delete Add_10_out; Add_10_out = NULL;
        delete BatchNorm_11_out; BatchNorm_11_out = NULL;
        delete Conv2D_35_out; Conv2D_35_out = NULL;
        delete Conv2D_36_out; Conv2D_36_out = NULL;
        delete Conv2D_37_out; Conv2D_37_out = NULL;
        delete Add_11_out; Add_11_out = NULL;
        delete BatchNorm_12_out; BatchNorm_12_out = NULL;
        delete Conv2D_38_out; Conv2D_38_out = NULL;
        delete Conv2D_39_out; Conv2D_39_out = NULL;
        delete Conv2D_40_out; Conv2D_40_out = NULL;
        delete Add_12_out; Add_12_out = NULL;
        delete BatchNorm_13_out; BatchNorm_13_out = NULL;
        delete Conv2D_41_out; Conv2D_41_out = NULL;
        delete Conv2D_42_out; Conv2D_42_out = NULL;
        delete Conv2D_43_out; Conv2D_43_out = NULL;
        delete Add_13_out; Add_13_out = NULL;
        delete BatchNorm_14_out; BatchNorm_14_out = NULL;
        delete Conv2D_44_out; Conv2D_44_out = NULL;
        delete Conv2D_45_out; Conv2D_45_out = NULL;
        delete Conv2D_46_out; Conv2D_46_out = NULL;
        delete Add_14_out; Add_14_out = NULL;
        delete BatchNorm_15_out; BatchNorm_15_out = NULL;
        delete Conv2D_47_out; Conv2D_47_out = NULL;
        delete Conv2D_48_out; Conv2D_48_out = NULL;
        delete Conv2D_49_out; Conv2D_49_out = NULL;
        delete Add_15_out; Add_15_out = NULL;
        delete BatchNorm_16_out; BatchNorm_16_out = NULL;
        delete Conv2D_50_out; Conv2D_50_out = NULL;
        delete Conv2D_51_out; Conv2D_51_out = NULL;
        delete Conv2D_52_out; Conv2D_52_out = NULL;
        delete Add_16_out; Add_16_out = NULL;
        delete BatchNorm_17_out; BatchNorm_17_out = NULL;
        delete Conv2D_53_out; Conv2D_53_out = NULL;
        delete Conv2D_54_out; Conv2D_54_out = NULL;
        delete Conv2D_55_out; Conv2D_55_out = NULL;
        delete Add_17_out; Add_17_out = NULL;
        delete BatchNorm_18_out; BatchNorm_18_out = NULL;
        delete Conv2D_56_out; Conv2D_56_out = NULL;
        delete Conv2D_57_out; Conv2D_57_out = NULL;
        delete Conv2D_58_out; Conv2D_58_out = NULL;
        delete Add_18_out; Add_18_out = NULL;
        delete BatchNorm_19_out; BatchNorm_19_out = NULL;
        delete Conv2D_59_out; Conv2D_59_out = NULL;
        delete Conv2D_60_out; Conv2D_60_out = NULL;
        delete Conv2D_61_out; Conv2D_61_out = NULL;
        delete Add_19_out; Add_19_out = NULL;
        delete BatchNorm_20_out; BatchNorm_20_out = NULL;
        delete Conv2D_62_out; Conv2D_62_out = NULL;
        delete Conv2D_63_out; Conv2D_63_out = NULL;
        delete Conv2D_64_out; Conv2D_64_out = NULL;
        delete Add_20_out; Add_20_out = NULL;
        delete BatchNorm_21_out; BatchNorm_21_out = NULL;
        delete Conv2D_65_out; Conv2D_65_out = NULL;
        delete Conv2D_66_out; Conv2D_66_out = NULL;
        delete Conv2D_67_out; Conv2D_67_out = NULL;
        delete Add_21_out; Add_21_out = NULL;
        delete BatchNorm_22_out; BatchNorm_22_out = NULL;
        delete Conv2D_68_out; Conv2D_68_out = NULL;
        delete Conv2D_69_out; Conv2D_69_out = NULL;
        delete Conv2D_70_out; Conv2D_70_out = NULL;
        delete Add_22_out; Add_22_out = NULL;
        delete BatchNorm_23_out; BatchNorm_23_out = NULL;
        delete Conv2D_71_out; Conv2D_71_out = NULL;
        delete Conv2D_72_out; Conv2D_72_out = NULL;
        delete Conv2D_73_out; Conv2D_73_out = NULL;
        delete Add_23_out; Add_23_out = NULL;
        delete BatchNorm_24_out; BatchNorm_24_out = NULL;
        delete Conv2D_74_out; Conv2D_74_out = NULL;
        delete Conv2D_75_out; Conv2D_75_out = NULL;
        delete Conv2D_76_out; Conv2D_76_out = NULL;
        delete Add_24_out; Add_24_out = NULL;
        delete BatchNorm_25_out; BatchNorm_25_out = NULL;
        delete Conv2D_77_out; Conv2D_77_out = NULL;
        delete Conv2D_78_out; Conv2D_78_out = NULL;
        delete Conv2D_79_out; Conv2D_79_out = NULL;
        delete Add_25_out; Add_25_out = NULL;
        delete BatchNorm_26_out; BatchNorm_26_out = NULL;
        delete Conv2D_80_out; Conv2D_80_out = NULL;
        delete Conv2D_81_out; Conv2D_81_out = NULL;
        delete Conv2D_82_out; Conv2D_82_out = NULL;
        delete Add_26_out; Add_26_out = NULL;
        delete BatchNorm_27_out; BatchNorm_27_out = NULL;
        delete Conv2D_83_out; Conv2D_83_out = NULL;
        delete Conv2D_84_out; Conv2D_84_out = NULL;
        delete Conv2D_85_out; Conv2D_85_out = NULL;
        delete Add_27_out; Add_27_out = NULL;
        delete BatchNorm_28_out; BatchNorm_28_out = NULL;
        delete Conv2D_86_out; Conv2D_86_out = NULL;
        delete Conv2D_87_out; Conv2D_87_out = NULL;
        delete Conv2D_88_out; Conv2D_88_out = NULL;
        delete Add_28_out; Add_28_out = NULL;
        delete BatchNorm_29_out; BatchNorm_29_out = NULL;
        delete Conv2D_89_out; Conv2D_89_out = NULL;
        delete Conv2D_90_out; Conv2D_90_out = NULL;
        delete Conv2D_91_out; Conv2D_91_out = NULL;
        delete Add_29_out; Add_29_out = NULL;
        delete BatchNorm_30_out; BatchNorm_30_out = NULL;
        delete Conv2D_92_out; Conv2D_92_out = NULL;
        delete Conv2D_93_out; Conv2D_93_out = NULL;
        delete Conv2D_94_out; Conv2D_94_out = NULL;
        delete Add_30_out; Add_30_out = NULL;
        delete BatchNorm_31_out; BatchNorm_31_out = NULL;
        delete Conv2D_95_out; Conv2D_95_out = NULL;
        delete Conv2D_96_out; Conv2D_96_out = NULL;
        delete Conv2D_97_out; Conv2D_97_out = NULL;
        delete Add_31_out; Add_31_out = NULL;
        delete BatchNorm_32_out; BatchNorm_32_out = NULL;
        delete Conv2D_98_out; Conv2D_98_out = NULL;
        delete Conv2D_99_out; Conv2D_99_out = NULL;
        delete Conv2D_100_out; Conv2D_100_out = NULL;
        delete Add_32_out; Add_32_out = NULL;
        delete BatchNorm_33_out; BatchNorm_33_out = NULL;
        delete Conv2D_101_out; Conv2D_101_out = NULL;
        delete Conv2D_102_out; Conv2D_102_out = NULL;
        delete Conv2D_103_out; Conv2D_103_out = NULL;
        delete Add_33_out; Add_33_out = NULL;
        delete BatchNorm_34_out; BatchNorm_34_out = NULL;
        delete Conv2D_104_out; Conv2D_104_out = NULL;
        delete Conv2D_105_out; Conv2D_105_out = NULL;
        delete Conv2D_106_out; Conv2D_106_out = NULL;
        delete Add_34_out; Add_34_out = NULL;
        delete BatchNorm_35_out; BatchNorm_35_out = NULL;
        delete Conv2D_107_out; Conv2D_107_out = NULL;
        delete Conv2D_108_out; Conv2D_108_out = NULL;
        delete Conv2D_109_out; Conv2D_109_out = NULL;
        delete Add_35_out; Add_35_out = NULL;
        delete BatchNorm_36_out; BatchNorm_36_out = NULL;
        delete Conv2D_110_out; Conv2D_110_out = NULL;
        delete Conv2D_111_out; Conv2D_111_out = NULL;
        delete Conv2D_112_out; Conv2D_112_out = NULL;
        delete Add_36_out; Add_36_out = NULL;
        delete BatchNorm_37_out; BatchNorm_37_out = NULL;
        delete Conv2D_113_out; Conv2D_113_out = NULL;
        delete Conv2D_114_out; Conv2D_114_out = NULL;
        delete Conv2D_115_out; Conv2D_115_out = NULL;
        delete Add_37_out; Add_37_out = NULL;
        delete BatchNorm_38_out; BatchNorm_38_out = NULL;
        delete Conv2D_116_out; Conv2D_116_out = NULL;
        delete Conv2D_117_out; Conv2D_117_out = NULL;
        delete Conv2D_118_out; Conv2D_118_out = NULL;
        delete Add_38_out; Add_38_out = NULL;
        delete BatchNorm_39_out; BatchNorm_39_out = NULL;
        delete Conv2D_119_out; Conv2D_119_out = NULL;
        delete Conv2D_120_out; Conv2D_120_out = NULL;
        delete Conv2D_121_out; Conv2D_121_out = NULL;
        delete Add_39_out; Add_39_out = NULL;
        delete BatchNorm_40_out; BatchNorm_40_out = NULL;
        delete Conv2D_122_out; Conv2D_122_out = NULL;
        delete Conv2D_123_out; Conv2D_123_out = NULL;
        delete Conv2D_124_out; Conv2D_124_out = NULL;
        delete Add_40_out; Add_40_out = NULL;
        delete BatchNorm_41_out; BatchNorm_41_out = NULL;
        delete Conv2D_125_out; Conv2D_125_out = NULL;
        delete Conv2D_126_out; Conv2D_126_out = NULL;
        delete Conv2D_127_out; Conv2D_127_out = NULL;
        delete Add_41_out; Add_41_out = NULL;
        delete BatchNorm_42_out; BatchNorm_42_out = NULL;
        delete Conv2D_128_out; Conv2D_128_out = NULL;
        delete Conv2D_129_out; Conv2D_129_out = NULL;
        delete Conv2D_130_out; Conv2D_130_out = NULL;
        delete Conv2D_131_out; Conv2D_131_out = NULL;
        delete Add_42_out; Add_42_out = NULL;
        delete BatchNorm_43_out; BatchNorm_43_out = NULL;
        delete Conv2D_132_out; Conv2D_132_out = NULL;
        delete Conv2D_133_out; Conv2D_133_out = NULL;
        delete Conv2D_134_out; Conv2D_134_out = NULL;
        delete Add_43_out; Add_43_out = NULL;
        delete BatchNorm_44_out; BatchNorm_44_out = NULL;
        delete AvgPool_1_out; AvgPool_1_out = NULL;
        delete Conv2D_135_out; Conv2D_135_out = NULL;
    }


    void release_weights() {
        delete[] Conv2D_1_w;
        delete[] Conv2D_1_b;
        delete[] Conv2D_2_w;
        delete[] Conv2D_2_b;
        delete[] Conv2D_3_w;
        delete[] Conv2D_3_b;
        delete[] Conv2D_4_w;
        delete[] Conv2D_4_b;
        delete[] Conv2D_5_w;
        delete[] Conv2D_5_b;
        delete[] Conv2D_6_w;
        delete[] Conv2D_6_b;
        delete[] Conv2D_7_w;
        delete[] Conv2D_7_b;
        delete[] Conv2D_8_w;
        delete[] Conv2D_8_b;
        delete[] Conv2D_9_w;
        delete[] Conv2D_9_b;
        delete[] Conv2D_10_w;
        delete[] Conv2D_10_b;
        delete[] Conv2D_11_w;
        delete[] Conv2D_11_b;
        delete[] Conv2D_12_w;
        delete[] Conv2D_12_b;
        delete[] Conv2D_13_w;
        delete[] Conv2D_13_b;
        delete[] Conv2D_14_w;
        delete[] Conv2D_14_b;
        delete[] Conv2D_15_w;
        delete[] Conv2D_15_b;
        delete[] Conv2D_16_w;
        delete[] Conv2D_16_b;
        delete[] Conv2D_17_w;
        delete[] Conv2D_17_b;
        delete[] Conv2D_18_w;
        delete[] Conv2D_18_b;
        delete[] Conv2D_19_w;
        delete[] Conv2D_19_b;
        delete[] Conv2D_20_w;
        delete[] Conv2D_20_b;
        delete[] Conv2D_21_w;
        delete[] Conv2D_21_b;
        delete[] Conv2D_22_w;
        delete[] Conv2D_22_b;
        delete[] Conv2D_23_w;
        delete[] Conv2D_23_b;
        delete[] Conv2D_24_w;
        delete[] Conv2D_24_b;
        delete[] Conv2D_25_w;
        delete[] Conv2D_25_b;
        delete[] Conv2D_26_w;
        delete[] Conv2D_26_b;
        delete[] Conv2D_27_w;
        delete[] Conv2D_27_b;
        delete[] Conv2D_28_w;
        delete[] Conv2D_28_b;
        delete[] Conv2D_29_w;
        delete[] Conv2D_29_b;
        delete[] Conv2D_30_w;
        delete[] Conv2D_30_b;
        delete[] Conv2D_31_w;
        delete[] Conv2D_31_b;
        delete[] Conv2D_32_w;
        delete[] Conv2D_32_b;
        delete[] Conv2D_33_w;
        delete[] Conv2D_33_b;
        delete[] Conv2D_34_w;
        delete[] Conv2D_34_b;
        delete[] Conv2D_35_w;
        delete[] Conv2D_35_b;
        delete[] Conv2D_36_w;
        delete[] Conv2D_36_b;
        delete[] Conv2D_37_w;
        delete[] Conv2D_37_b;
        delete[] Conv2D_38_w;
        delete[] Conv2D_38_b;
        delete[] Conv2D_39_w;
        delete[] Conv2D_39_b;
        delete[] Conv2D_40_w;
        delete[] Conv2D_40_b;
        delete[] Conv2D_41_w;
        delete[] Conv2D_41_b;
        delete[] Conv2D_42_w;
        delete[] Conv2D_42_b;
        delete[] Conv2D_43_w;
        delete[] Conv2D_43_b;
        delete[] Conv2D_44_w;
        delete[] Conv2D_44_b;
        delete[] Conv2D_45_w;
        delete[] Conv2D_45_b;
        delete[] Conv2D_46_w;
        delete[] Conv2D_46_b;
        delete[] Conv2D_47_w;
        delete[] Conv2D_47_b;
        delete[] Conv2D_48_w;
        delete[] Conv2D_48_b;
        delete[] Conv2D_49_w;
        delete[] Conv2D_49_b;
        delete[] Conv2D_50_w;
        delete[] Conv2D_50_b;
        delete[] Conv2D_51_w;
        delete[] Conv2D_51_b;
        delete[] Conv2D_52_w;
        delete[] Conv2D_52_b;
        delete[] Conv2D_53_w;
        delete[] Conv2D_53_b;
        delete[] Conv2D_54_w;
        delete[] Conv2D_54_b;
        delete[] Conv2D_55_w;
        delete[] Conv2D_55_b;
        delete[] Conv2D_56_w;
        delete[] Conv2D_56_b;
        delete[] Conv2D_57_w;
        delete[] Conv2D_57_b;
        delete[] Conv2D_58_w;
        delete[] Conv2D_58_b;
        delete[] Conv2D_59_w;
        delete[] Conv2D_59_b;
        delete[] Conv2D_60_w;
        delete[] Conv2D_60_b;
        delete[] Conv2D_61_w;
        delete[] Conv2D_61_b;
        delete[] Conv2D_62_w;
        delete[] Conv2D_62_b;
        delete[] Conv2D_63_w;
        delete[] Conv2D_63_b;
        delete[] Conv2D_64_w;
        delete[] Conv2D_64_b;
        delete[] Conv2D_65_w;
        delete[] Conv2D_65_b;
        delete[] Conv2D_66_w;
        delete[] Conv2D_66_b;
        delete[] Conv2D_67_w;
        delete[] Conv2D_67_b;
        delete[] Conv2D_68_w;
        delete[] Conv2D_68_b;
        delete[] Conv2D_69_w;
        delete[] Conv2D_69_b;
        delete[] Conv2D_70_w;
        delete[] Conv2D_70_b;
        delete[] Conv2D_71_w;
        delete[] Conv2D_71_b;
        delete[] Conv2D_72_w;
        delete[] Conv2D_72_b;
        delete[] Conv2D_73_w;
        delete[] Conv2D_73_b;
        delete[] Conv2D_74_w;
        delete[] Conv2D_74_b;
        delete[] Conv2D_75_w;
        delete[] Conv2D_75_b;
        delete[] Conv2D_76_w;
        delete[] Conv2D_76_b;
        delete[] Conv2D_77_w;
        delete[] Conv2D_77_b;
        delete[] Conv2D_78_w;
        delete[] Conv2D_78_b;
        delete[] Conv2D_79_w;
        delete[] Conv2D_79_b;
        delete[] Conv2D_80_w;
        delete[] Conv2D_80_b;
        delete[] Conv2D_81_w;
        delete[] Conv2D_81_b;
        delete[] Conv2D_82_w;
        delete[] Conv2D_82_b;
        delete[] Conv2D_83_w;
        delete[] Conv2D_83_b;
        delete[] Conv2D_84_w;
        delete[] Conv2D_84_b;
        delete[] Conv2D_85_w;
        delete[] Conv2D_85_b;
        delete[] Conv2D_86_w;
        delete[] Conv2D_86_b;
        delete[] Conv2D_87_w;
        delete[] Conv2D_87_b;
        delete[] Conv2D_88_w;
        delete[] Conv2D_88_b;
        delete[] Conv2D_89_w;
        delete[] Conv2D_89_b;
        delete[] Conv2D_90_w;
        delete[] Conv2D_90_b;
        delete[] Conv2D_91_w;
        delete[] Conv2D_91_b;
        delete[] Conv2D_92_w;
        delete[] Conv2D_92_b;
        delete[] Conv2D_93_w;
        delete[] Conv2D_93_b;
        delete[] Conv2D_94_w;
        delete[] Conv2D_94_b;
        delete[] Conv2D_95_w;
        delete[] Conv2D_95_b;
        delete[] Conv2D_96_w;
        delete[] Conv2D_96_b;
        delete[] Conv2D_97_w;
        delete[] Conv2D_97_b;
        delete[] Conv2D_98_w;
        delete[] Conv2D_98_b;
        delete[] Conv2D_99_w;
        delete[] Conv2D_99_b;
        delete[] Conv2D_100_w;
        delete[] Conv2D_100_b;
        delete[] Conv2D_101_w;
        delete[] Conv2D_101_b;
        delete[] Conv2D_102_w;
        delete[] Conv2D_102_b;
        delete[] Conv2D_103_w;
        delete[] Conv2D_103_b;
        delete[] Conv2D_104_w;
        delete[] Conv2D_104_b;
        delete[] Conv2D_105_w;
        delete[] Conv2D_105_b;
        delete[] Conv2D_106_w;
        delete[] Conv2D_106_b;
        delete[] Conv2D_107_w;
        delete[] Conv2D_107_b;
        delete[] Conv2D_108_w;
        delete[] Conv2D_108_b;
        delete[] Conv2D_109_w;
        delete[] Conv2D_109_b;
        delete[] Conv2D_110_w;
        delete[] Conv2D_110_b;
        delete[] Conv2D_111_w;
        delete[] Conv2D_111_b;
        delete[] Conv2D_112_w;
        delete[] Conv2D_112_b;
        delete[] Conv2D_113_w;
        delete[] Conv2D_113_b;
        delete[] Conv2D_114_w;
        delete[] Conv2D_114_b;
        delete[] Conv2D_115_w;
        delete[] Conv2D_115_b;
        delete[] Conv2D_116_w;
        delete[] Conv2D_116_b;
        delete[] Conv2D_117_w;
        delete[] Conv2D_117_b;
        delete[] Conv2D_118_w;
        delete[] Conv2D_118_b;
        delete[] Conv2D_119_w;
        delete[] Conv2D_119_b;
        delete[] Conv2D_120_w;
        delete[] Conv2D_120_b;
        delete[] Conv2D_121_w;
        delete[] Conv2D_121_b;
        delete[] Conv2D_122_w;
        delete[] Conv2D_122_b;
        delete[] Conv2D_123_w;
        delete[] Conv2D_123_b;
        delete[] Conv2D_124_w;
        delete[] Conv2D_124_b;
        delete[] Conv2D_125_w;
        delete[] Conv2D_125_b;
        delete[] Conv2D_126_w;
        delete[] Conv2D_126_b;
        delete[] Conv2D_127_w;
        delete[] Conv2D_127_b;
        delete[] Conv2D_128_w;
        delete[] Conv2D_128_b;
        delete[] Conv2D_129_w;
        delete[] Conv2D_129_b;
        delete[] Conv2D_130_w;
        delete[] Conv2D_130_b;
        delete[] Conv2D_131_w;
        delete[] Conv2D_131_b;
        delete[] Conv2D_132_w;
        delete[] Conv2D_132_b;
        delete[] Conv2D_133_w;
        delete[] Conv2D_133_b;
        delete[] Conv2D_134_w;
        delete[] Conv2D_134_b;
        delete[] Conv2D_135_w;
        delete[] Conv2D_135_b;
    }


    float* make_fake_input() {
        fake_input = new float[batch_size * input_height * input_width * input_channel];
        float *input_p = fake_input;

        for( int b = 0; b < batch_size; b ++ )
            for( int c = 0; c < input_channel; c ++ )
                for( int h = 0; h < input_height; h ++ )
                    for( int w = 0; w < input_width; w ++ )
                        *(input_p++) = 1.0;

        return fake_input;
    }


private:
    // Define basis variates
    float *fake_input;
    static int instance_num;
    const char *weights_path;
    int batch_size, input_height, input_width, input_channel;

    memory* src_memory = NULL;
    std::vector<primitive> net;
    engine cpu_engine = engine(engine::cpu, 0);

    // Define variates of net
    // Type                 Name       Origin_name
    Convolution*         Conv2D_1;	  // layer1/layer1_conv/Conv2D
    Pooling*             MaxPool_1;	  // layer2/MaxPool2D/MaxPool
    BatchNorm*           BatchNorm_1;	  // layer2/block0/common_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_2;	  // layer2/block0/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_3;	  // layer2/block0/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_4;	  // layer2/block0/sub3/sub3_conv/Conv2D
    Convolution*         Conv2D_5;	  // layer2/block0/shortcut/sub_sc/Conv2D
    Sum*                 Add_1;	  // layer2/block0/shortcut/add
    BatchNorm*           BatchNorm_2;	  // layer2/block1/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_6;	  // layer2/block1/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_7;	  // layer2/block1/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_8;	  // layer2/block1/sub3/sub3_conv/Conv2D
    Sum*                 Add_2;	  // layer2/block1/shortcut/add
    BatchNorm*           BatchNorm_3;	  // layer3/block0/common_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_9;	  // layer3/block0/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_10;	  // layer3/block0/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_11;	  // layer3/block0/sub3/sub3_conv/Conv2D
    Convolution*         Conv2D_12;	  // layer3/block0/shortcut/sub_sc/Conv2D
    Sum*                 Add_3;	  // layer3/block0/shortcut/add
    BatchNorm*           BatchNorm_4;	  // layer3/block1/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_13;	  // layer3/block1/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_14;	  // layer3/block1/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_15;	  // layer3/block1/sub3/sub3_conv/Conv2D
    Sum*                 Add_4;	  // layer3/block1/shortcut/add
    BatchNorm*           BatchNorm_5;	  // layer3/block2/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_16;	  // layer3/block2/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_17;	  // layer3/block2/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_18;	  // layer3/block2/sub3/sub3_conv/Conv2D
    Sum*                 Add_5;	  // layer3/block2/shortcut/add
    BatchNorm*           BatchNorm_6;	  // layer3/block3/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_19;	  // layer3/block3/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_20;	  // layer3/block3/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_21;	  // layer3/block3/sub3/sub3_conv/Conv2D
    Sum*                 Add_6;	  // layer3/block3/shortcut/add
    BatchNorm*           BatchNorm_7;	  // layer3/block4/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_22;	  // layer3/block4/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_23;	  // layer3/block4/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_24;	  // layer3/block4/sub3/sub3_conv/Conv2D
    Sum*                 Add_7;	  // layer3/block4/shortcut/add
    BatchNorm*           BatchNorm_8;	  // layer3/block5/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_25;	  // layer3/block5/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_26;	  // layer3/block5/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_27;	  // layer3/block5/sub3/sub3_conv/Conv2D
    Sum*                 Add_8;	  // layer3/block5/shortcut/add
    BatchNorm*           BatchNorm_9;	  // layer3/block6/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_28;	  // layer3/block6/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_29;	  // layer3/block6/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_30;	  // layer3/block6/sub3/sub3_conv/Conv2D
    Sum*                 Add_9;	  // layer3/block6/shortcut/add
    BatchNorm*           BatchNorm_10;	  // layer4/block0/common_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_31;	  // layer4/block0/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_32;	  // layer4/block0/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_33;	  // layer4/block0/sub3/sub3_conv/Conv2D
    Convolution*         Conv2D_34;	  // layer4/block0/shortcut/sub_sc/Conv2D
    Sum*                 Add_10;	  // layer4/block0/shortcut/add
    BatchNorm*           BatchNorm_11;	  // layer4/block1/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_35;	  // layer4/block1/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_36;	  // layer4/block1/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_37;	  // layer4/block1/sub3/sub3_conv/Conv2D
    Sum*                 Add_11;	  // layer4/block1/shortcut/add
    BatchNorm*           BatchNorm_12;	  // layer4/block2/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_38;	  // layer4/block2/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_39;	  // layer4/block2/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_40;	  // layer4/block2/sub3/sub3_conv/Conv2D
    Sum*                 Add_12;	  // layer4/block2/shortcut/add
    BatchNorm*           BatchNorm_13;	  // layer4/block3/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_41;	  // layer4/block3/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_42;	  // layer4/block3/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_43;	  // layer4/block3/sub3/sub3_conv/Conv2D
    Sum*                 Add_13;	  // layer4/block3/shortcut/add
    BatchNorm*           BatchNorm_14;	  // layer4/block4/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_44;	  // layer4/block4/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_45;	  // layer4/block4/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_46;	  // layer4/block4/sub3/sub3_conv/Conv2D
    Sum*                 Add_14;	  // layer4/block4/shortcut/add
    BatchNorm*           BatchNorm_15;	  // layer4/block5/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_47;	  // layer4/block5/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_48;	  // layer4/block5/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_49;	  // layer4/block5/sub3/sub3_conv/Conv2D
    Sum*                 Add_15;	  // layer4/block5/shortcut/add
    BatchNorm*           BatchNorm_16;	  // layer4/block6/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_50;	  // layer4/block6/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_51;	  // layer4/block6/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_52;	  // layer4/block6/sub3/sub3_conv/Conv2D
    Sum*                 Add_16;	  // layer4/block6/shortcut/add
    BatchNorm*           BatchNorm_17;	  // layer4/block7/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_53;	  // layer4/block7/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_54;	  // layer4/block7/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_55;	  // layer4/block7/sub3/sub3_conv/Conv2D
    Sum*                 Add_17;	  // layer4/block7/shortcut/add
    BatchNorm*           BatchNorm_18;	  // layer4/block8/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_56;	  // layer4/block8/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_57;	  // layer4/block8/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_58;	  // layer4/block8/sub3/sub3_conv/Conv2D
    Sum*                 Add_18;	  // layer4/block8/shortcut/add
    BatchNorm*           BatchNorm_19;	  // layer4/block9/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_59;	  // layer4/block9/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_60;	  // layer4/block9/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_61;	  // layer4/block9/sub3/sub3_conv/Conv2D
    Sum*                 Add_19;	  // layer4/block9/shortcut/add
    BatchNorm*           BatchNorm_20;	  // layer4/block10/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_62;	  // layer4/block10/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_63;	  // layer4/block10/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_64;	  // layer4/block10/sub3/sub3_conv/Conv2D
    Sum*                 Add_20;	  // layer4/block10/shortcut/add
    BatchNorm*           BatchNorm_21;	  // layer4/block11/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_65;	  // layer4/block11/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_66;	  // layer4/block11/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_67;	  // layer4/block11/sub3/sub3_conv/Conv2D
    Sum*                 Add_21;	  // layer4/block11/shortcut/add
    BatchNorm*           BatchNorm_22;	  // layer4/block12/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_68;	  // layer4/block12/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_69;	  // layer4/block12/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_70;	  // layer4/block12/sub3/sub3_conv/Conv2D
    Sum*                 Add_22;	  // layer4/block12/shortcut/add
    BatchNorm*           BatchNorm_23;	  // layer4/block13/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_71;	  // layer4/block13/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_72;	  // layer4/block13/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_73;	  // layer4/block13/sub3/sub3_conv/Conv2D
    Sum*                 Add_23;	  // layer4/block13/shortcut/add
    BatchNorm*           BatchNorm_24;	  // layer4/block14/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_74;	  // layer4/block14/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_75;	  // layer4/block14/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_76;	  // layer4/block14/sub3/sub3_conv/Conv2D
    Sum*                 Add_24;	  // layer4/block14/shortcut/add
    BatchNorm*           BatchNorm_25;	  // layer4/block15/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_77;	  // layer4/block15/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_78;	  // layer4/block15/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_79;	  // layer4/block15/sub3/sub3_conv/Conv2D
    Sum*                 Add_25;	  // layer4/block15/shortcut/add
    BatchNorm*           BatchNorm_26;	  // layer4/block16/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_80;	  // layer4/block16/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_81;	  // layer4/block16/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_82;	  // layer4/block16/sub3/sub3_conv/Conv2D
    Sum*                 Add_26;	  // layer4/block16/shortcut/add
    BatchNorm*           BatchNorm_27;	  // layer4/block17/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_83;	  // layer4/block17/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_84;	  // layer4/block17/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_85;	  // layer4/block17/sub3/sub3_conv/Conv2D
    Sum*                 Add_27;	  // layer4/block17/shortcut/add
    BatchNorm*           BatchNorm_28;	  // layer4/block18/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_86;	  // layer4/block18/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_87;	  // layer4/block18/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_88;	  // layer4/block18/sub3/sub3_conv/Conv2D
    Sum*                 Add_28;	  // layer4/block18/shortcut/add
    BatchNorm*           BatchNorm_29;	  // layer4/block19/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_89;	  // layer4/block19/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_90;	  // layer4/block19/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_91;	  // layer4/block19/sub3/sub3_conv/Conv2D
    Sum*                 Add_29;	  // layer4/block19/shortcut/add
    BatchNorm*           BatchNorm_30;	  // layer4/block20/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_92;	  // layer4/block20/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_93;	  // layer4/block20/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_94;	  // layer4/block20/sub3/sub3_conv/Conv2D
    Sum*                 Add_30;	  // layer4/block20/shortcut/add
    BatchNorm*           BatchNorm_31;	  // layer4/block21/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_95;	  // layer4/block21/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_96;	  // layer4/block21/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_97;	  // layer4/block21/sub3/sub3_conv/Conv2D
    Sum*                 Add_31;	  // layer4/block21/shortcut/add
    BatchNorm*           BatchNorm_32;	  // layer4/block22/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_98;	  // layer4/block22/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_99;	  // layer4/block22/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_100;	  // layer4/block22/sub3/sub3_conv/Conv2D
    Sum*                 Add_32;	  // layer4/block22/shortcut/add
    BatchNorm*           BatchNorm_33;	  // layer4/block23/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_101;	  // layer4/block23/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_102;	  // layer4/block23/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_103;	  // layer4/block23/sub3/sub3_conv/Conv2D
    Sum*                 Add_33;	  // layer4/block23/shortcut/add
    BatchNorm*           BatchNorm_34;	  // layer4/block24/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_104;	  // layer4/block24/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_105;	  // layer4/block24/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_106;	  // layer4/block24/sub3/sub3_conv/Conv2D
    Sum*                 Add_34;	  // layer4/block24/shortcut/add
    BatchNorm*           BatchNorm_35;	  // layer4/block25/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_107;	  // layer4/block25/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_108;	  // layer4/block25/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_109;	  // layer4/block25/sub3/sub3_conv/Conv2D
    Sum*                 Add_35;	  // layer4/block25/shortcut/add
    BatchNorm*           BatchNorm_36;	  // layer4/block26/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_110;	  // layer4/block26/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_111;	  // layer4/block26/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_112;	  // layer4/block26/sub3/sub3_conv/Conv2D
    Sum*                 Add_36;	  // layer4/block26/shortcut/add
    BatchNorm*           BatchNorm_37;	  // layer4/block27/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_113;	  // layer4/block27/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_114;	  // layer4/block27/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_115;	  // layer4/block27/sub3/sub3_conv/Conv2D
    Sum*                 Add_37;	  // layer4/block27/shortcut/add
    BatchNorm*           BatchNorm_38;	  // layer4/block28/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_116;	  // layer4/block28/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_117;	  // layer4/block28/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_118;	  // layer4/block28/sub3/sub3_conv/Conv2D
    Sum*                 Add_38;	  // layer4/block28/shortcut/add
    BatchNorm*           BatchNorm_39;	  // layer4/block29/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_119;	  // layer4/block29/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_120;	  // layer4/block29/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_121;	  // layer4/block29/sub3/sub3_conv/Conv2D
    Sum*                 Add_39;	  // layer4/block29/shortcut/add
    BatchNorm*           BatchNorm_40;	  // layer4/block30/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_122;	  // layer4/block30/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_123;	  // layer4/block30/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_124;	  // layer4/block30/sub3/sub3_conv/Conv2D
    Sum*                 Add_40;	  // layer4/block30/shortcut/add
    BatchNorm*           BatchNorm_41;	  // layer4/block31/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_125;	  // layer4/block31/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_126;	  // layer4/block31/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_127;	  // layer4/block31/sub3/sub3_conv/Conv2D
    Sum*                 Add_41;	  // layer4/block31/shortcut/add
    BatchNorm*           BatchNorm_42;	  // layer5/block0/common_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_128;	  // layer5/block0/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_129;	  // layer5/block0/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_130;	  // layer5/block0/sub3/sub3_conv/Conv2D
    Convolution*         Conv2D_131;	  // layer5/block0/shortcut/sub_sc/Conv2D
    Sum*                 Add_42;	  // layer5/block0/shortcut/add
    BatchNorm*           BatchNorm_43;	  // layer5/block1/residual_bn_relu/FusedBatchNorm
    Convolution*         Conv2D_132;	  // layer5/block1/sub1/sub1_conv/Conv2D
    Convolution*         Conv2D_133;	  // layer5/block1/sub2/sub2_conv/Conv2D
    Convolution*         Conv2D_134;	  // layer5/block1/sub3/sub3_conv/Conv2D
    Sum*                 Add_43;	  // layer5/block1/shortcut/add
    BatchNorm*           BatchNorm_44;	  // avg_fc/fc_bn/FusedBatchNorm
    Pooling*             AvgPool_1;	  // avg_fc/AvgPool
    Convolution*         Conv2D_135;	  // avg_fc/fc6/Conv2D

    // Define variates of outputs
    memory* Conv2D_1_out;
    memory* MaxPool_1_out;
    memory* BatchNorm_1_out;
    memory* Conv2D_2_out;
    memory* Conv2D_3_out;
    memory* Conv2D_4_out;
    memory* Conv2D_5_out;
    memory* Add_1_out;
    memory* BatchNorm_2_out;
    memory* Conv2D_6_out;
    memory* Conv2D_7_out;
    memory* Conv2D_8_out;
    memory* Add_2_out;
    memory* BatchNorm_3_out;
    memory* Conv2D_9_out;
    memory* Conv2D_10_out;
    memory* Conv2D_11_out;
    memory* Conv2D_12_out;
    memory* Add_3_out;
    memory* BatchNorm_4_out;
    memory* Conv2D_13_out;
    memory* Conv2D_14_out;
    memory* Conv2D_15_out;
    memory* Add_4_out;
    memory* BatchNorm_5_out;
    memory* Conv2D_16_out;
    memory* Conv2D_17_out;
    memory* Conv2D_18_out;
    memory* Add_5_out;
    memory* BatchNorm_6_out;
    memory* Conv2D_19_out;
    memory* Conv2D_20_out;
    memory* Conv2D_21_out;
    memory* Add_6_out;
    memory* BatchNorm_7_out;
    memory* Conv2D_22_out;
    memory* Conv2D_23_out;
    memory* Conv2D_24_out;
    memory* Add_7_out;
    memory* BatchNorm_8_out;
    memory* Conv2D_25_out;
    memory* Conv2D_26_out;
    memory* Conv2D_27_out;
    memory* Add_8_out;
    memory* BatchNorm_9_out;
    memory* Conv2D_28_out;
    memory* Conv2D_29_out;
    memory* Conv2D_30_out;
    memory* Add_9_out;
    memory* BatchNorm_10_out;
    memory* Conv2D_31_out;
    memory* Conv2D_32_out;
    memory* Conv2D_33_out;
    memory* Conv2D_34_out;
    memory* Add_10_out;
    memory* BatchNorm_11_out;
    memory* Conv2D_35_out;
    memory* Conv2D_36_out;
    memory* Conv2D_37_out;
    memory* Add_11_out;
    memory* BatchNorm_12_out;
    memory* Conv2D_38_out;
    memory* Conv2D_39_out;
    memory* Conv2D_40_out;
    memory* Add_12_out;
    memory* BatchNorm_13_out;
    memory* Conv2D_41_out;
    memory* Conv2D_42_out;
    memory* Conv2D_43_out;
    memory* Add_13_out;
    memory* BatchNorm_14_out;
    memory* Conv2D_44_out;
    memory* Conv2D_45_out;
    memory* Conv2D_46_out;
    memory* Add_14_out;
    memory* BatchNorm_15_out;
    memory* Conv2D_47_out;
    memory* Conv2D_48_out;
    memory* Conv2D_49_out;
    memory* Add_15_out;
    memory* BatchNorm_16_out;
    memory* Conv2D_50_out;
    memory* Conv2D_51_out;
    memory* Conv2D_52_out;
    memory* Add_16_out;
    memory* BatchNorm_17_out;
    memory* Conv2D_53_out;
    memory* Conv2D_54_out;
    memory* Conv2D_55_out;
    memory* Add_17_out;
    memory* BatchNorm_18_out;
    memory* Conv2D_56_out;
    memory* Conv2D_57_out;
    memory* Conv2D_58_out;
    memory* Add_18_out;
    memory* BatchNorm_19_out;
    memory* Conv2D_59_out;
    memory* Conv2D_60_out;
    memory* Conv2D_61_out;
    memory* Add_19_out;
    memory* BatchNorm_20_out;
    memory* Conv2D_62_out;
    memory* Conv2D_63_out;
    memory* Conv2D_64_out;
    memory* Add_20_out;
    memory* BatchNorm_21_out;
    memory* Conv2D_65_out;
    memory* Conv2D_66_out;
    memory* Conv2D_67_out;
    memory* Add_21_out;
    memory* BatchNorm_22_out;
    memory* Conv2D_68_out;
    memory* Conv2D_69_out;
    memory* Conv2D_70_out;
    memory* Add_22_out;
    memory* BatchNorm_23_out;
    memory* Conv2D_71_out;
    memory* Conv2D_72_out;
    memory* Conv2D_73_out;
    memory* Add_23_out;
    memory* BatchNorm_24_out;
    memory* Conv2D_74_out;
    memory* Conv2D_75_out;
    memory* Conv2D_76_out;
    memory* Add_24_out;
    memory* BatchNorm_25_out;
    memory* Conv2D_77_out;
    memory* Conv2D_78_out;
    memory* Conv2D_79_out;
    memory* Add_25_out;
    memory* BatchNorm_26_out;
    memory* Conv2D_80_out;
    memory* Conv2D_81_out;
    memory* Conv2D_82_out;
    memory* Add_26_out;
    memory* BatchNorm_27_out;
    memory* Conv2D_83_out;
    memory* Conv2D_84_out;
    memory* Conv2D_85_out;
    memory* Add_27_out;
    memory* BatchNorm_28_out;
    memory* Conv2D_86_out;
    memory* Conv2D_87_out;
    memory* Conv2D_88_out;
    memory* Add_28_out;
    memory* BatchNorm_29_out;
    memory* Conv2D_89_out;
    memory* Conv2D_90_out;
    memory* Conv2D_91_out;
    memory* Add_29_out;
    memory* BatchNorm_30_out;
    memory* Conv2D_92_out;
    memory* Conv2D_93_out;
    memory* Conv2D_94_out;
    memory* Add_30_out;
    memory* BatchNorm_31_out;
    memory* Conv2D_95_out;
    memory* Conv2D_96_out;
    memory* Conv2D_97_out;
    memory* Add_31_out;
    memory* BatchNorm_32_out;
    memory* Conv2D_98_out;
    memory* Conv2D_99_out;
    memory* Conv2D_100_out;
    memory* Add_32_out;
    memory* BatchNorm_33_out;
    memory* Conv2D_101_out;
    memory* Conv2D_102_out;
    memory* Conv2D_103_out;
    memory* Add_33_out;
    memory* BatchNorm_34_out;
    memory* Conv2D_104_out;
    memory* Conv2D_105_out;
    memory* Conv2D_106_out;
    memory* Add_34_out;
    memory* BatchNorm_35_out;
    memory* Conv2D_107_out;
    memory* Conv2D_108_out;
    memory* Conv2D_109_out;
    memory* Add_35_out;
    memory* BatchNorm_36_out;
    memory* Conv2D_110_out;
    memory* Conv2D_111_out;
    memory* Conv2D_112_out;
    memory* Add_36_out;
    memory* BatchNorm_37_out;
    memory* Conv2D_113_out;
    memory* Conv2D_114_out;
    memory* Conv2D_115_out;
    memory* Add_37_out;
    memory* BatchNorm_38_out;
    memory* Conv2D_116_out;
    memory* Conv2D_117_out;
    memory* Conv2D_118_out;
    memory* Add_38_out;
    memory* BatchNorm_39_out;
    memory* Conv2D_119_out;
    memory* Conv2D_120_out;
    memory* Conv2D_121_out;
    memory* Add_39_out;
    memory* BatchNorm_40_out;
    memory* Conv2D_122_out;
    memory* Conv2D_123_out;
    memory* Conv2D_124_out;
    memory* Add_40_out;
    memory* BatchNorm_41_out;
    memory* Conv2D_125_out;
    memory* Conv2D_126_out;
    memory* Conv2D_127_out;
    memory* Add_41_out;
    memory* BatchNorm_42_out;
    memory* Conv2D_128_out;
    memory* Conv2D_129_out;
    memory* Conv2D_130_out;
    memory* Conv2D_131_out;
    memory* Add_42_out;
    memory* BatchNorm_43_out;
    memory* Conv2D_132_out;
    memory* Conv2D_133_out;
    memory* Conv2D_134_out;
    memory* Add_43_out;
    memory* BatchNorm_44_out;
    memory* AvgPool_1_out;
    memory* Conv2D_135_out;

    // Define variates of weights
    static float *Conv2D_1_w, *Conv2D_1_b;
    static float *BatchNorm_1_weights, *BatchNorm_1_mean, *BatchNorm_1_variance;
    static float *Conv2D_2_w, *Conv2D_2_b;
    static float *Conv2D_3_w, *Conv2D_3_b;
    static float *Conv2D_4_w, *Conv2D_4_b;
    static float *Conv2D_5_w, *Conv2D_5_b;
    static float *BatchNorm_2_weights, *BatchNorm_2_mean, *BatchNorm_2_variance;
    static float *Conv2D_6_w, *Conv2D_6_b;
    static float *Conv2D_7_w, *Conv2D_7_b;
    static float *Conv2D_8_w, *Conv2D_8_b;
    static float *BatchNorm_3_weights, *BatchNorm_3_mean, *BatchNorm_3_variance;
    static float *Conv2D_9_w, *Conv2D_9_b;
    static float *Conv2D_10_w, *Conv2D_10_b;
    static float *Conv2D_11_w, *Conv2D_11_b;
    static float *Conv2D_12_w, *Conv2D_12_b;
    static float *BatchNorm_4_weights, *BatchNorm_4_mean, *BatchNorm_4_variance;
    static float *Conv2D_13_w, *Conv2D_13_b;
    static float *Conv2D_14_w, *Conv2D_14_b;
    static float *Conv2D_15_w, *Conv2D_15_b;
    static float *BatchNorm_5_weights, *BatchNorm_5_mean, *BatchNorm_5_variance;
    static float *Conv2D_16_w, *Conv2D_16_b;
    static float *Conv2D_17_w, *Conv2D_17_b;
    static float *Conv2D_18_w, *Conv2D_18_b;
    static float *BatchNorm_6_weights, *BatchNorm_6_mean, *BatchNorm_6_variance;
    static float *Conv2D_19_w, *Conv2D_19_b;
    static float *Conv2D_20_w, *Conv2D_20_b;
    static float *Conv2D_21_w, *Conv2D_21_b;
    static float *BatchNorm_7_weights, *BatchNorm_7_mean, *BatchNorm_7_variance;
    static float *Conv2D_22_w, *Conv2D_22_b;
    static float *Conv2D_23_w, *Conv2D_23_b;
    static float *Conv2D_24_w, *Conv2D_24_b;
    static float *BatchNorm_8_weights, *BatchNorm_8_mean, *BatchNorm_8_variance;
    static float *Conv2D_25_w, *Conv2D_25_b;
    static float *Conv2D_26_w, *Conv2D_26_b;
    static float *Conv2D_27_w, *Conv2D_27_b;
    static float *BatchNorm_9_weights, *BatchNorm_9_mean, *BatchNorm_9_variance;
    static float *Conv2D_28_w, *Conv2D_28_b;
    static float *Conv2D_29_w, *Conv2D_29_b;
    static float *Conv2D_30_w, *Conv2D_30_b;
    static float *BatchNorm_10_weights, *BatchNorm_10_mean, *BatchNorm_10_variance;
    static float *Conv2D_31_w, *Conv2D_31_b;
    static float *Conv2D_32_w, *Conv2D_32_b;
    static float *Conv2D_33_w, *Conv2D_33_b;
    static float *Conv2D_34_w, *Conv2D_34_b;
    static float *BatchNorm_11_weights, *BatchNorm_11_mean, *BatchNorm_11_variance;
    static float *Conv2D_35_w, *Conv2D_35_b;
    static float *Conv2D_36_w, *Conv2D_36_b;
    static float *Conv2D_37_w, *Conv2D_37_b;
    static float *BatchNorm_12_weights, *BatchNorm_12_mean, *BatchNorm_12_variance;
    static float *Conv2D_38_w, *Conv2D_38_b;
    static float *Conv2D_39_w, *Conv2D_39_b;
    static float *Conv2D_40_w, *Conv2D_40_b;
    static float *BatchNorm_13_weights, *BatchNorm_13_mean, *BatchNorm_13_variance;
    static float *Conv2D_41_w, *Conv2D_41_b;
    static float *Conv2D_42_w, *Conv2D_42_b;
    static float *Conv2D_43_w, *Conv2D_43_b;
    static float *BatchNorm_14_weights, *BatchNorm_14_mean, *BatchNorm_14_variance;
    static float *Conv2D_44_w, *Conv2D_44_b;
    static float *Conv2D_45_w, *Conv2D_45_b;
    static float *Conv2D_46_w, *Conv2D_46_b;
    static float *BatchNorm_15_weights, *BatchNorm_15_mean, *BatchNorm_15_variance;
    static float *Conv2D_47_w, *Conv2D_47_b;
    static float *Conv2D_48_w, *Conv2D_48_b;
    static float *Conv2D_49_w, *Conv2D_49_b;
    static float *BatchNorm_16_weights, *BatchNorm_16_mean, *BatchNorm_16_variance;
    static float *Conv2D_50_w, *Conv2D_50_b;
    static float *Conv2D_51_w, *Conv2D_51_b;
    static float *Conv2D_52_w, *Conv2D_52_b;
    static float *BatchNorm_17_weights, *BatchNorm_17_mean, *BatchNorm_17_variance;
    static float *Conv2D_53_w, *Conv2D_53_b;
    static float *Conv2D_54_w, *Conv2D_54_b;
    static float *Conv2D_55_w, *Conv2D_55_b;
    static float *BatchNorm_18_weights, *BatchNorm_18_mean, *BatchNorm_18_variance;
    static float *Conv2D_56_w, *Conv2D_56_b;
    static float *Conv2D_57_w, *Conv2D_57_b;
    static float *Conv2D_58_w, *Conv2D_58_b;
    static float *BatchNorm_19_weights, *BatchNorm_19_mean, *BatchNorm_19_variance;
    static float *Conv2D_59_w, *Conv2D_59_b;
    static float *Conv2D_60_w, *Conv2D_60_b;
    static float *Conv2D_61_w, *Conv2D_61_b;
    static float *BatchNorm_20_weights, *BatchNorm_20_mean, *BatchNorm_20_variance;
    static float *Conv2D_62_w, *Conv2D_62_b;
    static float *Conv2D_63_w, *Conv2D_63_b;
    static float *Conv2D_64_w, *Conv2D_64_b;
    static float *BatchNorm_21_weights, *BatchNorm_21_mean, *BatchNorm_21_variance;
    static float *Conv2D_65_w, *Conv2D_65_b;
    static float *Conv2D_66_w, *Conv2D_66_b;
    static float *Conv2D_67_w, *Conv2D_67_b;
    static float *BatchNorm_22_weights, *BatchNorm_22_mean, *BatchNorm_22_variance;
    static float *Conv2D_68_w, *Conv2D_68_b;
    static float *Conv2D_69_w, *Conv2D_69_b;
    static float *Conv2D_70_w, *Conv2D_70_b;
    static float *BatchNorm_23_weights, *BatchNorm_23_mean, *BatchNorm_23_variance;
    static float *Conv2D_71_w, *Conv2D_71_b;
    static float *Conv2D_72_w, *Conv2D_72_b;
    static float *Conv2D_73_w, *Conv2D_73_b;
    static float *BatchNorm_24_weights, *BatchNorm_24_mean, *BatchNorm_24_variance;
    static float *Conv2D_74_w, *Conv2D_74_b;
    static float *Conv2D_75_w, *Conv2D_75_b;
    static float *Conv2D_76_w, *Conv2D_76_b;
    static float *BatchNorm_25_weights, *BatchNorm_25_mean, *BatchNorm_25_variance;
    static float *Conv2D_77_w, *Conv2D_77_b;
    static float *Conv2D_78_w, *Conv2D_78_b;
    static float *Conv2D_79_w, *Conv2D_79_b;
    static float *BatchNorm_26_weights, *BatchNorm_26_mean, *BatchNorm_26_variance;
    static float *Conv2D_80_w, *Conv2D_80_b;
    static float *Conv2D_81_w, *Conv2D_81_b;
    static float *Conv2D_82_w, *Conv2D_82_b;
    static float *BatchNorm_27_weights, *BatchNorm_27_mean, *BatchNorm_27_variance;
    static float *Conv2D_83_w, *Conv2D_83_b;
    static float *Conv2D_84_w, *Conv2D_84_b;
    static float *Conv2D_85_w, *Conv2D_85_b;
    static float *BatchNorm_28_weights, *BatchNorm_28_mean, *BatchNorm_28_variance;
    static float *Conv2D_86_w, *Conv2D_86_b;
    static float *Conv2D_87_w, *Conv2D_87_b;
    static float *Conv2D_88_w, *Conv2D_88_b;
    static float *BatchNorm_29_weights, *BatchNorm_29_mean, *BatchNorm_29_variance;
    static float *Conv2D_89_w, *Conv2D_89_b;
    static float *Conv2D_90_w, *Conv2D_90_b;
    static float *Conv2D_91_w, *Conv2D_91_b;
    static float *BatchNorm_30_weights, *BatchNorm_30_mean, *BatchNorm_30_variance;
    static float *Conv2D_92_w, *Conv2D_92_b;
    static float *Conv2D_93_w, *Conv2D_93_b;
    static float *Conv2D_94_w, *Conv2D_94_b;
    static float *BatchNorm_31_weights, *BatchNorm_31_mean, *BatchNorm_31_variance;
    static float *Conv2D_95_w, *Conv2D_95_b;
    static float *Conv2D_96_w, *Conv2D_96_b;
    static float *Conv2D_97_w, *Conv2D_97_b;
    static float *BatchNorm_32_weights, *BatchNorm_32_mean, *BatchNorm_32_variance;
    static float *Conv2D_98_w, *Conv2D_98_b;
    static float *Conv2D_99_w, *Conv2D_99_b;
    static float *Conv2D_100_w, *Conv2D_100_b;
    static float *BatchNorm_33_weights, *BatchNorm_33_mean, *BatchNorm_33_variance;
    static float *Conv2D_101_w, *Conv2D_101_b;
    static float *Conv2D_102_w, *Conv2D_102_b;
    static float *Conv2D_103_w, *Conv2D_103_b;
    static float *BatchNorm_34_weights, *BatchNorm_34_mean, *BatchNorm_34_variance;
    static float *Conv2D_104_w, *Conv2D_104_b;
    static float *Conv2D_105_w, *Conv2D_105_b;
    static float *Conv2D_106_w, *Conv2D_106_b;
    static float *BatchNorm_35_weights, *BatchNorm_35_mean, *BatchNorm_35_variance;
    static float *Conv2D_107_w, *Conv2D_107_b;
    static float *Conv2D_108_w, *Conv2D_108_b;
    static float *Conv2D_109_w, *Conv2D_109_b;
    static float *BatchNorm_36_weights, *BatchNorm_36_mean, *BatchNorm_36_variance;
    static float *Conv2D_110_w, *Conv2D_110_b;
    static float *Conv2D_111_w, *Conv2D_111_b;
    static float *Conv2D_112_w, *Conv2D_112_b;
    static float *BatchNorm_37_weights, *BatchNorm_37_mean, *BatchNorm_37_variance;
    static float *Conv2D_113_w, *Conv2D_113_b;
    static float *Conv2D_114_w, *Conv2D_114_b;
    static float *Conv2D_115_w, *Conv2D_115_b;
    static float *BatchNorm_38_weights, *BatchNorm_38_mean, *BatchNorm_38_variance;
    static float *Conv2D_116_w, *Conv2D_116_b;
    static float *Conv2D_117_w, *Conv2D_117_b;
    static float *Conv2D_118_w, *Conv2D_118_b;
    static float *BatchNorm_39_weights, *BatchNorm_39_mean, *BatchNorm_39_variance;
    static float *Conv2D_119_w, *Conv2D_119_b;
    static float *Conv2D_120_w, *Conv2D_120_b;
    static float *Conv2D_121_w, *Conv2D_121_b;
    static float *BatchNorm_40_weights, *BatchNorm_40_mean, *BatchNorm_40_variance;
    static float *Conv2D_122_w, *Conv2D_122_b;
    static float *Conv2D_123_w, *Conv2D_123_b;
    static float *Conv2D_124_w, *Conv2D_124_b;
    static float *BatchNorm_41_weights, *BatchNorm_41_mean, *BatchNorm_41_variance;
    static float *Conv2D_125_w, *Conv2D_125_b;
    static float *Conv2D_126_w, *Conv2D_126_b;
    static float *Conv2D_127_w, *Conv2D_127_b;
    static float *BatchNorm_42_weights, *BatchNorm_42_mean, *BatchNorm_42_variance;
    static float *Conv2D_128_w, *Conv2D_128_b;
    static float *Conv2D_129_w, *Conv2D_129_b;
    static float *Conv2D_130_w, *Conv2D_130_b;
    static float *Conv2D_131_w, *Conv2D_131_b;
    static float *BatchNorm_43_weights, *BatchNorm_43_mean, *BatchNorm_43_variance;
    static float *Conv2D_132_w, *Conv2D_132_b;
    static float *Conv2D_133_w, *Conv2D_133_b;
    static float *Conv2D_134_w, *Conv2D_134_b;
    static float *BatchNorm_44_weights, *BatchNorm_44_mean, *BatchNorm_44_variance;
    static float *Conv2D_135_w, *Conv2D_135_b;
};

int Model::instance_num = 0;
// Create variates of weights
float* Model::Conv2D_1_w = new float[7 * 7 * 64 * 3];
float* Model::Conv2D_1_b = new float[64];
float* Model::BatchNorm_1_weights = new float[64 * 2];
float* Model::BatchNorm_1_mean = new float[64];
float* Model::BatchNorm_1_variance = new float[64];
float* Model::Conv2D_2_w = new float[1 * 1 * 64 * 64];
float* Model::Conv2D_2_b = new float[64];
float* Model::Conv2D_3_w = new float[3 * 3 * 64 * 64];
float* Model::Conv2D_3_b = new float[64];
float* Model::Conv2D_4_w = new float[1 * 1 * 256 * 64];
float* Model::Conv2D_4_b = new float[256];
float* Model::Conv2D_5_w = new float[1 * 1 * 256 * 64];
float* Model::Conv2D_5_b = new float[256];
float* Model::BatchNorm_2_weights = new float[512 * 2];
float* Model::BatchNorm_2_mean = new float[512];
float* Model::BatchNorm_2_variance = new float[512];
float* Model::Conv2D_6_w = new float[1 * 1 * 64 * 512];
float* Model::Conv2D_6_b = new float[64];
float* Model::Conv2D_7_w = new float[3 * 3 * 64 * 64];
float* Model::Conv2D_7_b = new float[64];
float* Model::Conv2D_8_w = new float[1 * 1 * 256 * 64];
float* Model::Conv2D_8_b = new float[256];
float* Model::BatchNorm_3_weights = new float[768 * 2];
float* Model::BatchNorm_3_mean = new float[768];
float* Model::BatchNorm_3_variance = new float[768];
float* Model::Conv2D_9_w = new float[1 * 1 * 128 * 768];
float* Model::Conv2D_9_b = new float[128];
float* Model::Conv2D_10_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_10_b = new float[128];
float* Model::Conv2D_11_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_11_b = new float[512];
float* Model::Conv2D_12_w = new float[1 * 1 * 512 * 768];
float* Model::Conv2D_12_b = new float[512];
float* Model::BatchNorm_4_weights = new float[1024 * 2];
float* Model::BatchNorm_4_mean = new float[1024];
float* Model::BatchNorm_4_variance = new float[1024];
float* Model::Conv2D_13_w = new float[1 * 1 * 128 * 1024];
float* Model::Conv2D_13_b = new float[128];
float* Model::Conv2D_14_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_14_b = new float[128];
float* Model::Conv2D_15_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_15_b = new float[512];
float* Model::BatchNorm_5_weights = new float[1536 * 2];
float* Model::BatchNorm_5_mean = new float[1536];
float* Model::BatchNorm_5_variance = new float[1536];
float* Model::Conv2D_16_w = new float[1 * 1 * 128 * 1536];
float* Model::Conv2D_16_b = new float[128];
float* Model::Conv2D_17_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_17_b = new float[128];
float* Model::Conv2D_18_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_18_b = new float[512];
float* Model::BatchNorm_6_weights = new float[2048 * 2];
float* Model::BatchNorm_6_mean = new float[2048];
float* Model::BatchNorm_6_variance = new float[2048];
float* Model::Conv2D_19_w = new float[1 * 1 * 128 * 2048];
float* Model::Conv2D_19_b = new float[128];
float* Model::Conv2D_20_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_20_b = new float[128];
float* Model::Conv2D_21_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_21_b = new float[512];
float* Model::BatchNorm_7_weights = new float[2560 * 2];
float* Model::BatchNorm_7_mean = new float[2560];
float* Model::BatchNorm_7_variance = new float[2560];
float* Model::Conv2D_22_w = new float[1 * 1 * 128 * 2560];
float* Model::Conv2D_22_b = new float[128];
float* Model::Conv2D_23_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_23_b = new float[128];
float* Model::Conv2D_24_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_24_b = new float[512];
float* Model::BatchNorm_8_weights = new float[3072 * 2];
float* Model::BatchNorm_8_mean = new float[3072];
float* Model::BatchNorm_8_variance = new float[3072];
float* Model::Conv2D_25_w = new float[1 * 1 * 128 * 3072];
float* Model::Conv2D_25_b = new float[128];
float* Model::Conv2D_26_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_26_b = new float[128];
float* Model::Conv2D_27_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_27_b = new float[512];
float* Model::BatchNorm_9_weights = new float[3584 * 2];
float* Model::BatchNorm_9_mean = new float[3584];
float* Model::BatchNorm_9_variance = new float[3584];
float* Model::Conv2D_28_w = new float[1 * 1 * 128 * 3584];
float* Model::Conv2D_28_b = new float[128];
float* Model::Conv2D_29_w = new float[3 * 3 * 128 * 128];
float* Model::Conv2D_29_b = new float[128];
float* Model::Conv2D_30_w = new float[1 * 1 * 512 * 128];
float* Model::Conv2D_30_b = new float[512];
float* Model::BatchNorm_10_weights = new float[4096 * 2];
float* Model::BatchNorm_10_mean = new float[4096];
float* Model::BatchNorm_10_variance = new float[4096];
float* Model::Conv2D_31_w = new float[1 * 1 * 256 * 4096];
float* Model::Conv2D_31_b = new float[256];
float* Model::Conv2D_32_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_32_b = new float[256];
float* Model::Conv2D_33_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_33_b = new float[1024];
float* Model::Conv2D_34_w = new float[1 * 1 * 1024 * 4096];
float* Model::Conv2D_34_b = new float[1024];
float* Model::BatchNorm_11_weights = new float[2048 * 2];
float* Model::BatchNorm_11_mean = new float[2048];
float* Model::BatchNorm_11_variance = new float[2048];
float* Model::Conv2D_35_w = new float[1 * 1 * 256 * 2048];
float* Model::Conv2D_35_b = new float[256];
float* Model::Conv2D_36_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_36_b = new float[256];
float* Model::Conv2D_37_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_37_b = new float[1024];
float* Model::BatchNorm_12_weights = new float[3072 * 2];
float* Model::BatchNorm_12_mean = new float[3072];
float* Model::BatchNorm_12_variance = new float[3072];
float* Model::Conv2D_38_w = new float[1 * 1 * 256 * 3072];
float* Model::Conv2D_38_b = new float[256];
float* Model::Conv2D_39_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_39_b = new float[256];
float* Model::Conv2D_40_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_40_b = new float[1024];
float* Model::BatchNorm_13_weights = new float[4096 * 2];
float* Model::BatchNorm_13_mean = new float[4096];
float* Model::BatchNorm_13_variance = new float[4096];
float* Model::Conv2D_41_w = new float[1 * 1 * 256 * 4096];
float* Model::Conv2D_41_b = new float[256];
float* Model::Conv2D_42_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_42_b = new float[256];
float* Model::Conv2D_43_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_43_b = new float[1024];
float* Model::BatchNorm_14_weights = new float[5120 * 2];
float* Model::BatchNorm_14_mean = new float[5120];
float* Model::BatchNorm_14_variance = new float[5120];
float* Model::Conv2D_44_w = new float[1 * 1 * 256 * 5120];
float* Model::Conv2D_44_b = new float[256];
float* Model::Conv2D_45_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_45_b = new float[256];
float* Model::Conv2D_46_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_46_b = new float[1024];
float* Model::BatchNorm_15_weights = new float[6144 * 2];
float* Model::BatchNorm_15_mean = new float[6144];
float* Model::BatchNorm_15_variance = new float[6144];
float* Model::Conv2D_47_w = new float[1 * 1 * 256 * 6144];
float* Model::Conv2D_47_b = new float[256];
float* Model::Conv2D_48_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_48_b = new float[256];
float* Model::Conv2D_49_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_49_b = new float[1024];
float* Model::BatchNorm_16_weights = new float[7168 * 2];
float* Model::BatchNorm_16_mean = new float[7168];
float* Model::BatchNorm_16_variance = new float[7168];
float* Model::Conv2D_50_w = new float[1 * 1 * 256 * 7168];
float* Model::Conv2D_50_b = new float[256];
float* Model::Conv2D_51_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_51_b = new float[256];
float* Model::Conv2D_52_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_52_b = new float[1024];
float* Model::BatchNorm_17_weights = new float[8192 * 2];
float* Model::BatchNorm_17_mean = new float[8192];
float* Model::BatchNorm_17_variance = new float[8192];
float* Model::Conv2D_53_w = new float[1 * 1 * 256 * 8192];
float* Model::Conv2D_53_b = new float[256];
float* Model::Conv2D_54_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_54_b = new float[256];
float* Model::Conv2D_55_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_55_b = new float[1024];
float* Model::BatchNorm_18_weights = new float[9216 * 2];
float* Model::BatchNorm_18_mean = new float[9216];
float* Model::BatchNorm_18_variance = new float[9216];
float* Model::Conv2D_56_w = new float[1 * 1 * 256 * 9216];
float* Model::Conv2D_56_b = new float[256];
float* Model::Conv2D_57_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_57_b = new float[256];
float* Model::Conv2D_58_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_58_b = new float[1024];
float* Model::BatchNorm_19_weights = new float[10240 * 2];
float* Model::BatchNorm_19_mean = new float[10240];
float* Model::BatchNorm_19_variance = new float[10240];
float* Model::Conv2D_59_w = new float[1 * 1 * 256 * 10240];
float* Model::Conv2D_59_b = new float[256];
float* Model::Conv2D_60_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_60_b = new float[256];
float* Model::Conv2D_61_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_61_b = new float[1024];
float* Model::BatchNorm_20_weights = new float[11264 * 2];
float* Model::BatchNorm_20_mean = new float[11264];
float* Model::BatchNorm_20_variance = new float[11264];
float* Model::Conv2D_62_w = new float[1 * 1 * 256 * 11264];
float* Model::Conv2D_62_b = new float[256];
float* Model::Conv2D_63_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_63_b = new float[256];
float* Model::Conv2D_64_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_64_b = new float[1024];
float* Model::BatchNorm_21_weights = new float[12288 * 2];
float* Model::BatchNorm_21_mean = new float[12288];
float* Model::BatchNorm_21_variance = new float[12288];
float* Model::Conv2D_65_w = new float[1 * 1 * 256 * 12288];
float* Model::Conv2D_65_b = new float[256];
float* Model::Conv2D_66_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_66_b = new float[256];
float* Model::Conv2D_67_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_67_b = new float[1024];
float* Model::BatchNorm_22_weights = new float[13312 * 2];
float* Model::BatchNorm_22_mean = new float[13312];
float* Model::BatchNorm_22_variance = new float[13312];
float* Model::Conv2D_68_w = new float[1 * 1 * 256 * 13312];
float* Model::Conv2D_68_b = new float[256];
float* Model::Conv2D_69_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_69_b = new float[256];
float* Model::Conv2D_70_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_70_b = new float[1024];
float* Model::BatchNorm_23_weights = new float[14336 * 2];
float* Model::BatchNorm_23_mean = new float[14336];
float* Model::BatchNorm_23_variance = new float[14336];
float* Model::Conv2D_71_w = new float[1 * 1 * 256 * 14336];
float* Model::Conv2D_71_b = new float[256];
float* Model::Conv2D_72_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_72_b = new float[256];
float* Model::Conv2D_73_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_73_b = new float[1024];
float* Model::BatchNorm_24_weights = new float[15360 * 2];
float* Model::BatchNorm_24_mean = new float[15360];
float* Model::BatchNorm_24_variance = new float[15360];
float* Model::Conv2D_74_w = new float[1 * 1 * 256 * 15360];
float* Model::Conv2D_74_b = new float[256];
float* Model::Conv2D_75_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_75_b = new float[256];
float* Model::Conv2D_76_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_76_b = new float[1024];
float* Model::BatchNorm_25_weights = new float[16384 * 2];
float* Model::BatchNorm_25_mean = new float[16384];
float* Model::BatchNorm_25_variance = new float[16384];
float* Model::Conv2D_77_w = new float[1 * 1 * 256 * 16384];
float* Model::Conv2D_77_b = new float[256];
float* Model::Conv2D_78_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_78_b = new float[256];
float* Model::Conv2D_79_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_79_b = new float[1024];
float* Model::BatchNorm_26_weights = new float[17408 * 2];
float* Model::BatchNorm_26_mean = new float[17408];
float* Model::BatchNorm_26_variance = new float[17408];
float* Model::Conv2D_80_w = new float[1 * 1 * 256 * 17408];
float* Model::Conv2D_80_b = new float[256];
float* Model::Conv2D_81_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_81_b = new float[256];
float* Model::Conv2D_82_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_82_b = new float[1024];
float* Model::BatchNorm_27_weights = new float[18432 * 2];
float* Model::BatchNorm_27_mean = new float[18432];
float* Model::BatchNorm_27_variance = new float[18432];
float* Model::Conv2D_83_w = new float[1 * 1 * 256 * 18432];
float* Model::Conv2D_83_b = new float[256];
float* Model::Conv2D_84_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_84_b = new float[256];
float* Model::Conv2D_85_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_85_b = new float[1024];
float* Model::BatchNorm_28_weights = new float[19456 * 2];
float* Model::BatchNorm_28_mean = new float[19456];
float* Model::BatchNorm_28_variance = new float[19456];
float* Model::Conv2D_86_w = new float[1 * 1 * 256 * 19456];
float* Model::Conv2D_86_b = new float[256];
float* Model::Conv2D_87_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_87_b = new float[256];
float* Model::Conv2D_88_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_88_b = new float[1024];
float* Model::BatchNorm_29_weights = new float[20480 * 2];
float* Model::BatchNorm_29_mean = new float[20480];
float* Model::BatchNorm_29_variance = new float[20480];
float* Model::Conv2D_89_w = new float[1 * 1 * 256 * 20480];
float* Model::Conv2D_89_b = new float[256];
float* Model::Conv2D_90_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_90_b = new float[256];
float* Model::Conv2D_91_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_91_b = new float[1024];
float* Model::BatchNorm_30_weights = new float[21504 * 2];
float* Model::BatchNorm_30_mean = new float[21504];
float* Model::BatchNorm_30_variance = new float[21504];
float* Model::Conv2D_92_w = new float[1 * 1 * 256 * 21504];
float* Model::Conv2D_92_b = new float[256];
float* Model::Conv2D_93_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_93_b = new float[256];
float* Model::Conv2D_94_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_94_b = new float[1024];
float* Model::BatchNorm_31_weights = new float[22528 * 2];
float* Model::BatchNorm_31_mean = new float[22528];
float* Model::BatchNorm_31_variance = new float[22528];
float* Model::Conv2D_95_w = new float[1 * 1 * 256 * 22528];
float* Model::Conv2D_95_b = new float[256];
float* Model::Conv2D_96_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_96_b = new float[256];
float* Model::Conv2D_97_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_97_b = new float[1024];
float* Model::BatchNorm_32_weights = new float[23552 * 2];
float* Model::BatchNorm_32_mean = new float[23552];
float* Model::BatchNorm_32_variance = new float[23552];
float* Model::Conv2D_98_w = new float[1 * 1 * 256 * 23552];
float* Model::Conv2D_98_b = new float[256];
float* Model::Conv2D_99_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_99_b = new float[256];
float* Model::Conv2D_100_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_100_b = new float[1024];
float* Model::BatchNorm_33_weights = new float[24576 * 2];
float* Model::BatchNorm_33_mean = new float[24576];
float* Model::BatchNorm_33_variance = new float[24576];
float* Model::Conv2D_101_w = new float[1 * 1 * 256 * 24576];
float* Model::Conv2D_101_b = new float[256];
float* Model::Conv2D_102_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_102_b = new float[256];
float* Model::Conv2D_103_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_103_b = new float[1024];
float* Model::BatchNorm_34_weights = new float[25600 * 2];
float* Model::BatchNorm_34_mean = new float[25600];
float* Model::BatchNorm_34_variance = new float[25600];
float* Model::Conv2D_104_w = new float[1 * 1 * 256 * 25600];
float* Model::Conv2D_104_b = new float[256];
float* Model::Conv2D_105_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_105_b = new float[256];
float* Model::Conv2D_106_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_106_b = new float[1024];
float* Model::BatchNorm_35_weights = new float[26624 * 2];
float* Model::BatchNorm_35_mean = new float[26624];
float* Model::BatchNorm_35_variance = new float[26624];
float* Model::Conv2D_107_w = new float[1 * 1 * 256 * 26624];
float* Model::Conv2D_107_b = new float[256];
float* Model::Conv2D_108_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_108_b = new float[256];
float* Model::Conv2D_109_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_109_b = new float[1024];
float* Model::BatchNorm_36_weights = new float[27648 * 2];
float* Model::BatchNorm_36_mean = new float[27648];
float* Model::BatchNorm_36_variance = new float[27648];
float* Model::Conv2D_110_w = new float[1 * 1 * 256 * 27648];
float* Model::Conv2D_110_b = new float[256];
float* Model::Conv2D_111_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_111_b = new float[256];
float* Model::Conv2D_112_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_112_b = new float[1024];
float* Model::BatchNorm_37_weights = new float[28672 * 2];
float* Model::BatchNorm_37_mean = new float[28672];
float* Model::BatchNorm_37_variance = new float[28672];
float* Model::Conv2D_113_w = new float[1 * 1 * 256 * 28672];
float* Model::Conv2D_113_b = new float[256];
float* Model::Conv2D_114_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_114_b = new float[256];
float* Model::Conv2D_115_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_115_b = new float[1024];
float* Model::BatchNorm_38_weights = new float[29696 * 2];
float* Model::BatchNorm_38_mean = new float[29696];
float* Model::BatchNorm_38_variance = new float[29696];
float* Model::Conv2D_116_w = new float[1 * 1 * 256 * 29696];
float* Model::Conv2D_116_b = new float[256];
float* Model::Conv2D_117_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_117_b = new float[256];
float* Model::Conv2D_118_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_118_b = new float[1024];
float* Model::BatchNorm_39_weights = new float[30720 * 2];
float* Model::BatchNorm_39_mean = new float[30720];
float* Model::BatchNorm_39_variance = new float[30720];
float* Model::Conv2D_119_w = new float[1 * 1 * 256 * 30720];
float* Model::Conv2D_119_b = new float[256];
float* Model::Conv2D_120_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_120_b = new float[256];
float* Model::Conv2D_121_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_121_b = new float[1024];
float* Model::BatchNorm_40_weights = new float[31744 * 2];
float* Model::BatchNorm_40_mean = new float[31744];
float* Model::BatchNorm_40_variance = new float[31744];
float* Model::Conv2D_122_w = new float[1 * 1 * 256 * 31744];
float* Model::Conv2D_122_b = new float[256];
float* Model::Conv2D_123_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_123_b = new float[256];
float* Model::Conv2D_124_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_124_b = new float[1024];
float* Model::BatchNorm_41_weights = new float[32768 * 2];
float* Model::BatchNorm_41_mean = new float[32768];
float* Model::BatchNorm_41_variance = new float[32768];
float* Model::Conv2D_125_w = new float[1 * 1 * 256 * 32768];
float* Model::Conv2D_125_b = new float[256];
float* Model::Conv2D_126_w = new float[3 * 3 * 256 * 256];
float* Model::Conv2D_126_b = new float[256];
float* Model::Conv2D_127_w = new float[1 * 1 * 1024 * 256];
float* Model::Conv2D_127_b = new float[1024];
float* Model::BatchNorm_42_weights = new float[33792 * 2];
float* Model::BatchNorm_42_mean = new float[33792];
float* Model::BatchNorm_42_variance = new float[33792];
float* Model::Conv2D_128_w = new float[1 * 1 * 512 * 33792];
float* Model::Conv2D_128_b = new float[512];
float* Model::Conv2D_129_w = new float[3 * 3 * 512 * 512];
float* Model::Conv2D_129_b = new float[512];
float* Model::Conv2D_130_w = new float[1 * 1 * 2048 * 512];
float* Model::Conv2D_130_b = new float[2048];
float* Model::Conv2D_131_w = new float[1 * 1 * 2048 * 33792];
float* Model::Conv2D_131_b = new float[2048];
float* Model::BatchNorm_43_weights = new float[4096 * 2];
float* Model::BatchNorm_43_mean = new float[4096];
float* Model::BatchNorm_43_variance = new float[4096];
float* Model::Conv2D_132_w = new float[1 * 1 * 512 * 4096];
float* Model::Conv2D_132_b = new float[512];
float* Model::Conv2D_133_w = new float[3 * 3 * 512 * 512];
float* Model::Conv2D_133_b = new float[512];
float* Model::Conv2D_134_w = new float[1 * 1 * 2048 * 512];
float* Model::Conv2D_134_b = new float[2048];
float* Model::BatchNorm_44_weights = new float[6144 * 2];
float* Model::BatchNorm_44_mean = new float[6144];
float* Model::BatchNorm_44_variance = new float[6144];
float* Model::Conv2D_135_w = new float[1 * 1 * 6 * 6144];
float* Model::Conv2D_135_b = new float[6];


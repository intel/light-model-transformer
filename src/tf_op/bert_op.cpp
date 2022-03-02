// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "bert_layer_quant_int8.h"

#include <vector>

using namespace tensorflow;

REGISTER_OP("Bert")
    .Input("embedded: float")
    .Input("input_mask: MaskT")
    .Input("weights: NumWeights * float")
    .Output("encoded: float")
    .Attr("MaskT: {int32, int64, float, double}")
    .Attr("NumWeights: int >= 16") // num_layers = NumWeights/16
    .Attr("HiddenSize: int = 768")
    .Attr("NumAttentionHeads: int = 12")
    .Attr("IntermediateSize: int = 3072")
    .Attr("HiddenAct: string = 'gelu_tanh'");

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

template <typename MaskT>
class BertOp : public OpKernel
{
public:
    explicit BertOp(OpKernelConstruction *context) : OpKernel(context)
    {
        printf("Bert op Construction!\n");

        int num;
        OP_REQUIRES_OK(context, context->GetAttr("NumWeights", &num));
        OP_REQUIRES_OK(context, context->GetAttr("HiddenSize", &hiddenSize));

        // Each layer has 16 weights
        this->layers = num / 16;
        this->bert_layers.reserve(this->layers);

        for (int i = 0; i < this->layers; ++i)
        {
            auto t = new BertLayer(ctx);
            this->bert_layers.push_back(t);
        }

        this->initialized = false;
    }

    ~BertOp()
    {
        for (int i = 0; i < this->bert_layers.size(); ++i)
        {
            delete this->bert_layers[i];
        }
        this->bert_layers.clear();
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &tensor_embeded = context->input(0);
        const Tensor &tensor_masks = context->input(1);

        OP_REQUIRES(context, (tensor_embeded.dims() == 2 || tensor_embeded.dims() == 3),
                    errors::InvalidArgument("Dims unexpected: dim(input) != 2"));

        // TF2 models provide batched input, first dimension is batch size, required to be 1 for now.
        if (tensor_embeded.dims() == 3)
        {
            OP_REQUIRES(context, (tensor_embeded.dim_size(0) == 1),
                        errors::InvalidArgument("Only batch size of 1 is supported right now."));
        }

        // Due to the above, for a 2D tensor: dim_size(1) == ctx.hiddensize,
        // and for 3D tensor dim_size(2) == ctx.hiddensize
        int hidden_size_dim_idx = tensor_embeded.dims() - 1;
        OP_REQUIRES(context, (tensor_embeded.dim_size(hidden_size_dim_idx) == ctx.hiddenSize),
                    errors::InvalidArgument("Unexpected hidden size"));


        float *embeded = (float *)tensor_embeded.tensor_data().data();
        MaskT *masks = (MaskT *)tensor_masks.tensor_data().data();

        int total_tokens_idx = tensor_embeded.dims() - 2;
        int total_tokens = tensor_embeded.dim_size(total_tokens_idx); // total_tokens = batch_size * tokens_each_def128

        // Initialize the weights and mode
        if (!initialized)
        {
            // TODO: protect it
            initWeights(context);
            initialized = true;
        }

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, tensor_embeded.shape(),
                                    &output_tensor));
        float *output = (float *)output_tensor->tensor_data().data();


        ctx.setInputMask(masks);
        dnnl::memory::dims dims{total_tokens, hiddenSize};
        auto pinput = dnnl_wrappers::AttachMemory(ctx.dnnl_context.getEngine(), dims, embeded, false);
        for (int i = 0; i < this->layers; ++i)
        {
            this->bert_layers[i]->forward(pinput);
        }

        // Copy data to output
        memcpy(output, embeded, sizeof(float) * hiddenSize * total_tokens);
    }

private:
    void initWeights(OpKernelContext *context)
    {
        int idx = 2;
        for (int i = 0; i < this->bert_layers.size(); ++i)
        {
            float *queryW = (float *)context->input(idx++).tensor_data().data();
            float *queryB = (float *)context->input(idx++).tensor_data().data();
            float *keyW = (float *)context->input(idx++).tensor_data().data();
            float *keyB = (float *)context->input(idx++).tensor_data().data();
            float *valueW = (float *)context->input(idx++).tensor_data().data();
            float *valueB = (float *)context->input(idx++).tensor_data().data();

            float *att_dense_w = (float *)context->input(idx++).tensor_data().data();
            float *att_dense_b = (float *)context->input(idx++).tensor_data().data();

            float *gamma1 = (float *)context->input(idx++).tensor_data().data();
            float *beta1 = (float *)context->input(idx++).tensor_data().data();

            float *intermediateW = (float *)context->input(idx++).tensor_data().data();
            float *intermediateB = (float *)context->input(idx++).tensor_data().data();

            float *outputW = (float *)context->input(idx++).tensor_data().data();
            float *outputB = (float *)context->input(idx++).tensor_data().data();

            float *gamma2 = (float *)context->input(idx++).tensor_data().data();
            float *beta2 = (float *)context->input(idx++).tensor_data().data();

            float *minmax = frozen_minmax[i];

            this->bert_layers[i]->setWeights(queryW, queryB,
                                             keyW, keyB,
                                             valueW, valueB,
                                             att_dense_w, att_dense_b,
                                             gamma1, beta1,
                                             intermediateW, intermediateB,
                                             outputW, outputB,
                                             gamma2, beta2,
                                             minmax);
        }
    }

private:
    int layers;
    int hiddenSize;

    BertContext ctx;
    std::vector<BertLayer *> bert_layers;

    bool initialized;
    float frozen_minmax [12][8] = {
        {-10.85244083404541015625, 4.14164829254150390625, -1.6212508678436279296875, 2.18305110931396484375, -64.5349578857421875, 9.17784881591796875, -0.16926576197147369384765625, 12.69039154052734375},
        {-10.01922702789306640625, 3.2598330974578857421875, -2.52011966705322265625, 3.17220592498779296875, -70.322662353515625, 4.564808368682861328125, -0.16925294697284698486328125, 10.93472957611083984375},
        {-11.37454319000244140625, 4.04611110687255859375, -2.5044767856597900390625, 3.4310567378997802734375, -56.21540069580078125, 5.208764553070068359375, -0.16948534548282623291015625, 72.20577239990234375},
        {-14.79791736602783203125, 4.259090423583984375, -2.8403589725494384765625, 3.91925144195556640625, -93.42563629150390625, 5.099577426910400390625, -0.1689991652965545654296875, 9.5706195831298828125},
        {-13.21285343170166015625, 4.449753284454345703125, -3.1772515773773193359375, 4.3330135345458984375, -101.334869384765625, 5.41256046295166015625, -0.16838109493255615234375, 10.64498996734619140625},
        {-13.93945217132568359375, 5.1448192596435546875, -2.5481836795806884765625, 3.48368167877197265625, -91.05278778076171875, 5.9057769775390625, -0.16948328912258148193359375, 12.6811923980712890625},
        {-14.12649059295654296875, 5.23845577239990234375, -2.814735889434814453125, 3.2215893268585205078125, -89.623870849609375, 6.68107700347900390625, -0.16898013651371002197265625, 11.01731777191162109375},
        {-13.5746974945068359375, 4.71494960784912109375, -2.7004568576812744140625, 3.2631299495697021484375, -87.90279388427734375, 7.388260364532470703125, -0.16951541602611541748046875, 8.03197765350341796875},
        {-15.597011566162109375, 6.920653820037841796875, -3.0222375392913818359375, 3.777666568756103515625, -83.6142730712890625, 10.2494525909423828125, -0.1686449944972991943359375, 23.9402790069580078125},
        {-15.88373565673828125, 10.81757640838623046875, -2.6777179241180419921875, 3.3885133266448974609375, -48.061458587646484375, 16.7345333099365234375, -0.156786620616912841796875, 92.52396392822265625},
        {-18.6183719635009765625, 11.54715251922607421875, -2.11896610260009765625, 3.066336154937744140625, -41.8497314453125, 19.4496479034423828125, -0.16698478162288665771484375, 141.4157867431640625},
        {-23.8061676025390625, 11.55181217193603515625, -2.552584171295166015625, 3.7034885883331298828125, -36.45532989501953125, 16.997623443603515625, -0.16963402926921844482421875, 8.112117767333984375},
    };
};

REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<int32>("MaskT"), BertOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<int64>("MaskT"), BertOp<int64>);
REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<float>("MaskT"), BertOp<float>);
REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<double>("MaskT"), BertOp<double>);

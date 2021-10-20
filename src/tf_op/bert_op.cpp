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
    .Attr("MaskT: {int32, int64}")
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

        OP_REQUIRES(context, (tensor_embeded.dims() == 2),
                    errors::InvalidArgument("Dims unexpected: dim(input) != 2"));
        OP_REQUIRES(context, (tensor_embeded.dim_size(1) == ctx.hiddenSize),
                    errors::InvalidArgument("Unexpected hidden size"));

        float *embeded = (float *)tensor_embeded.tensor_data().data();
        MaskT *masks = (MaskT *)tensor_masks.tensor_data().data();

        int total_tokens = tensor_embeded.dim_size(0); // total_tokens = batch_size * tokens_each_def128

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

        // Wrap input into matrix
        hpj::Matrix<float> input_buf(embeded, total_tokens, hiddenSize, hiddenSize);
        hpj::Matrix<float> *pinput = &input_buf;

        ctx.setInputMask(masks);
        for (int i = 0; i < this->layers; ++i)
        {
            hpj::Matrix<float> &out = this->bert_layers[i]->forward(*pinput, 0);
            pinput = &out;
        }

        // Copy data to output
#pragma omp parallel for
        for (int i = 0; i < total_tokens; ++i)
        {
            memcpy(output + i * hiddenSize, pinput->Row(i), sizeof(float) * hiddenSize);
        }
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

            this->bert_layers[i]->setWeights(queryW, queryB,
                                             keyW, keyB,
                                             valueW, valueB,
                                             att_dense_w, att_dense_b,
                                             gamma1, beta1,
                                             intermediateW, intermediateB,
                                             outputW, outputB,
                                             gamma2, beta2);
        }
    }

private:
    int layers;
    int hiddenSize;

    BertContext ctx;
    std::vector<BertLayer *> bert_layers;

    bool initialized;
};

REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<int32>("MaskT"), BertOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<int64>("MaskT"), BertOp<int64>);
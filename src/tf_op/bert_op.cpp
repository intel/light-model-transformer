// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

#include "bert_layer_quant_int8.h"
#include "bert_type_traits.h"
#include "dnnl_common.h"
#include "tensor_adapter.h"

#include "dnnl.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>
#include <stdexcept>

using namespace tensorflow;

REGISTER_OP("Bert")
    .Input("embedded: float")
    .Input("input_mask: MaskT")
    .Input("weights: NumWeights * float")
    .Output("encoded: float")
    .Attr("MaskT: {int32, int64, float, double}")
    .Attr("QuantizableDataType: {float, qint8}")        // These types are arbitrary, the kernel builder macro calls
    .Attr("NonQuantizableDataType: {float, bfloat16}")  // translate these into booleans for the BertOp.
    // .Attr("UseQuantization: bool = false")   // The boolean attrs are more readable,
    // .Attr("UseBFloat16: bool = false")       // but don't seem to work in TF 1.15.4
    .Attr("NumWeights: int >= 16") // num_layers = NumWeights/16
    .Attr("HiddenSize: int = 768")
    .Attr("NumAttentionHeads: int = 12")
    .Attr("IntermediateSize: int = 3072")
    .Attr("HiddenAct: {'gelu_tanh'} = 'gelu_tanh'");

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)



template <bool UseQuantization = false, bool UseBFloat16 = false>
class BertOp : public OpKernel
{
    using InputT = typename use_quantization<UseQuantization>::type;
    using BatchInputT = typename use_bfloat16<UseBFloat16>::type;
    using BertContextT = BertContext<InputT, BatchInputT>;

public:
    explicit BertOp(OpKernelConstruction* context)
        : OpKernel{context}
        , bert_ctx_builder{}
        , ctx{}
        , bert_layers{}
        , tensor_adapter{}
        , initialized{false}

    {
        try
        {
            int num_weights;
            OP_REQUIRES_OK(context, context->GetAttr("NumWeights", &num_weights));
            
            OP_REQUIRES(context, num_weights % BertContextT::tensors_per_layer == 0,
                errors::InvalidArgument("NumWeights must be a multiple of BertLayer::weights_per_layer"));

            int hidden_size;
            OP_REQUIRES_OK(context, context->GetAttr("HiddenSize", &hidden_size));

            int intermediate_size;
            OP_REQUIRES_OK(context, context->GetAttr("IntermediateSize", &intermediate_size));

            int num_attention_heads;
            OP_REQUIRES_OK(context, context->GetAttr("NumAttentionHeads", &num_attention_heads));

            OP_REQUIRES(context, num_attention_heads * BertContextT::head_size == hidden_size,
                errors::InvalidArgument("Constraint not met: HiddenSize = NumAttentionHead * HeadSize"));

            int layers = num_weights / BertContextT::tensors_per_layer;

            OP_REQUIRES(context, layers == 12, errors::InvalidArgument("Currently only BERT-base with 12 layers is supported."));

            bert_ctx_builder.NumLayers(layers);
            bert_ctx_builder.HiddenSize(hidden_size);
            bert_ctx_builder.IntermediateSize(intermediate_size);
            bert_ctx_builder.MaxTokenSize(max_token_size);
            bert_ctx_builder.NumAttentionHeads(num_attention_heads);


            {
                std::stringstream ss;
                ss << std::boolalpha << "BertOp<" << UseQuantization << ", " << UseBFloat16 << "> constructed.";
                std::cout << ss.str() << std::endl;
            }
        }
        catch(std::exception& ex)
        {
            OP_REQUIRES(context, false, errors::Unknown(ex.what()));
        }
        catch(...)
        {
            OP_REQUIRES(context, false, errors::Unknown("Unknown error during Bert op construction."));
        }
    }

    void Compute(OpKernelContext *context) override
    {
        try
        {
            const Tensor &tensor_embedded = context->input(0);
            const Tensor &tensor_masks = context->input(1);

            {
                int batch_size = getBatchSize(tensor_embedded);
                OP_REQUIRES(context, batch_size > 0, errors::InvalidArgument("Failed to get batch size."));
                bert_ctx_builder.BatchSize(batch_size);
                tensor_adapter.Init(batch_size, bert_ctx_builder.MaxTokenSize(), bert_ctx_builder.HiddenSize(),
                                    bert_ctx_builder.NumAttentionHeads(), bert_ctx_builder.IntermediateSize());
            }



            // Validate tensor dimensions.
            // The macros won't work properly if they are in a separate function.
            {
                OP_REQUIRES(context, (tensor_masks.dims() == 3),
                    errors::InvalidArgument("The mask tensor must have 3 dimensions."));

                OP_REQUIRES(context, (tensor_embedded.dims() == 2 || tensor_embedded.dims() == 3),
                            errors::InvalidArgument("The input tensor must have either 2 or 3 dimensions."));

                if (tensor_embedded.dims() == 2)
                {
                    OP_REQUIRES(context, (tensor_embedded.dim_size(0) % max_token_size == 0),
                        errors::InvalidArgument("2D input must be divisible by max_token_size on the first dimension."));
                }

                // For a 2D tensor: dim_size(1) == ctx->hiddensize,
                // For 3D tensor: dim_size(2) == ctx->hiddensize
                int hidden_size_dim_idx = tensor_embedded.dims() - 1;
                OP_REQUIRES(context, (tensor_embedded.dim_size(hidden_size_dim_idx) == bert_ctx_builder.HiddenSize()),
                            errors::InvalidArgument("Unexpected hidden size"));
            }

            // Initialize the BertContext, BertLayers and weights
            {
                auto status = initializeIfNeeded(context);
                OP_REQUIRES_OK(context, status);
            }

            // Fail if batch size changed.
            {
                int batch_size = getBatchSize(tensor_embedded);
                assert(batch_size > 0);
                OP_REQUIRES(context, batch_size == ctx->batch_, errors::InvalidArgument("Batch size changed unexpectedly."));
            }

            OP_REQUIRES_OK(context, setInputMask(tensor_masks));

            auto embeddings = tensor_adapter.AsDnnlMemory(tensor_embedded, ctx->dnnl_context.getEngine());
            try
            {
                for (const auto& bert_layer : this->bert_layers)
                {
                    bert_layer->forward(embeddings);
                }
            }
            catch (const dnnl::error& e)
            {
                std::string message = "Forward computation failed. what(): ";
                message += e.what();
                OP_REQUIRES(context, false, errors::Internal(message));
            }

            // Copy data to output
            Tensor output_tensor;
            output_tensor = tensor_embedded;
            OP_REQUIRES(context, output_tensor.IsInitialized(),
                        errors::InvalidArgument("Failed to set output tensor."));
            context->set_output(0, output_tensor);
        }
        catch(std::exception& ex)
        {
            OP_REQUIRES(context, false, errors::Unknown(ex.what()));
        }
        catch(...)
        {
            OP_REQUIRES(context, false, errors::Unknown("Unknown error during Bert op compute function."));
        }
    }

private:


    tensorflow::Status initializeIfNeeded(OpKernelContext* context)
    {
        // BertContext must be initialized before BertLayers and weights,
        // so that we can verify the dimensions of the input tensor.
        // TODO: Reinitialize BertContext, and BertLayers if batch size changes.
        if (!initialized)
        {
            try
            {
                // We store all initialized objects in temporaries, so that we don't need to undo initialization if any
                // of these fails. Any exception will be caught, and RAII will handle cleanup of these temporaries.
                auto tmp_ctx = initBertContext();
                auto tmp_layers = initLayers(tmp_ctx);
                auto tmp_tensors = storeTensors(context);
                initWeights(tmp_tensors, tmp_layers, tmp_ctx);

                static_assert(std::is_nothrow_move_assignable<decltype(tmp_ctx)>::value,
                    "BertContext must be no-throw move assignable.");
                static_assert(std::is_nothrow_move_assignable<decltype(tmp_layers)>::value,
                    "BertLayer container must be no-throw move assignable.");
                static_assert(std::is_nothrow_move_assignable<decltype(tmp_tensors)>::value,
                    "Tensor container must be no-throw move assignable.");
                
                // At this point everything is initialized, so we move-assign it to members.
                // The static_asserts above ensure this is noexcept, so again, no need to undo
                // partial initialization in case of failure.
                this->ctx = std::move(tmp_ctx);
                this->bert_layer_tensors = std::move(tmp_tensors);
                this->bert_layers = std::move(tmp_layers);
                
                initialized = true;
            }
            catch(const std::exception& e)
            {
                initialized = false;
                return errors::Aborted(e.what());
            }
        }
        return Status::OK();
    }

    std::shared_ptr<BertContextT> initBertContext()
    {
        auto tmp_ctx = bert_ctx_builder.Build<InputT, BatchInputT>();
        std::cout << "BertContext initialized: maxTokenSize = " << tmp_ctx->maxTokenSize
                  << ", hiddenSize = " << tmp_ctx->hiddenSize << ", intermediateSize = " << tmp_ctx->intermediateSize
                  << ", batch = " << tmp_ctx->batch_ << ", numLayers = " << tmp_ctx->numLayers << std::endl;
        return tmp_ctx;
    }

    /**
     * @brief Determine the batch size from tensor dimensions.
     * 
     * For a 3D tensor, batch size is the first dimension.
     * For a 2D tensor, the first dimension is assumed to be batch_size*max_token_size
     * 
     * @param tensor The tensor, must be 2D or 3D
     * @return int - The batch size
     */
    int getBatchSize(const Tensor& tensor)
    {
        // TF2 models can provide batched input, so they have one extra dimension.
        if (tensor.dims() == 3)
        {
            return tensor.dim_size(0);
        }
        else if (tensor.dims() == 2)
        {
            // OP_REQUIRES in Compute() guarantees that
            // tensor.dim_size(0) % max_token_size == 0
            return tensor.dim_size(0) / max_token_size;
        }

        // Will not happen due to OP_REQUIRES checks in Compute(),
        // but may be used for paranoid asserts in debug builds.
        return -1;
    }

    std::vector<std::unique_ptr<BertLayer<BertContextT>>> initLayers(const std::shared_ptr<BertContextT>& tmp_ctx)
    {
        std::vector<std::unique_ptr<BertLayer<BertContextT>>> tmp_layers;
        tmp_layers.reserve(tmp_ctx->numLayers);

        for (int i = 0; i < tmp_ctx->numLayers; ++i)
        {
            tmp_layers.emplace_back(std::make_unique<BertLayer<BertContextT>>(tmp_ctx));
        }
        return tmp_layers;
    }

    std::vector<tensorflow::Tensor> storeTensors(OpKernelContext* context)
    {
        std::vector<tensorflow::Tensor> tensors;
        for(int i = 2; i < context->num_inputs(); ++i)
        {
            tensorflow::Tensor t;
            if (!t.CopyFrom(context->input(i), context->input(i).shape()))
            {
                throw std::runtime_error("Failed to store weight tensor.");
            }
            tensors.emplace_back(std::move(t));
        }
        return tensors;
    }

    void initWeights(const std::vector<tensorflow::Tensor>& tensors,
                     std::vector<std::unique_ptr<BertLayer<BertContextT>>>& layers,
                     const std::shared_ptr<BertContextT> bert_context)
    {
        auto& engine = bert_context->dnnl_context.getEngine();

        auto tensor_at = [&tensors](size_t layer, size_t offset)
        {
            return tensors.at(BertContextT::tensors_per_layer * layer + offset);
        };

        for (size_t i = 0; i < layers.size(); ++i)
        {
            dnnl::memory queryW = tensor_adapter.AsDnnlMemory(tensor_at(i, 0), engine);
            dnnl::memory queryB = tensor_adapter.AsDnnlMemory(tensor_at(i, 1), engine);
            
            dnnl::memory keyW = tensor_adapter.AsDnnlMemory(tensor_at(i, 2), engine);
            dnnl::memory keyB = tensor_adapter.AsDnnlMemory(tensor_at(i, 3), engine);

            dnnl::memory valueW = tensor_adapter.AsDnnlMemory(tensor_at(i, 4), engine);
            dnnl::memory valueB = tensor_adapter.AsDnnlMemory(tensor_at(i, 5), engine);

            dnnl::memory att_dense_w = tensor_adapter.AsDnnlMemory(tensor_at(i, 6), engine);
            dnnl::memory att_dense_b = tensor_adapter.AsDnnlMemory(tensor_at(i, 7), engine);

            dnnl::memory gamma1 = tensor_adapter.AsDnnlMemory(tensor_at(i, 8), engine);
            dnnl::memory beta1 = tensor_adapter.AsDnnlMemory(tensor_at(i, 9), engine);

            dnnl::memory intermediateW = tensor_adapter.AsDnnlMemory(tensor_at(i, 10), engine);
            dnnl::memory intermediateB = tensor_adapter.AsDnnlMemory(tensor_at(i, 11), engine);

            dnnl::memory outputW = tensor_adapter.AsDnnlMemory(tensor_at(i, 12), engine);
            dnnl::memory outputB = tensor_adapter.AsDnnlMemory(tensor_at(i, 13), engine);

            dnnl::memory gamma2 = tensor_adapter.AsDnnlMemory(tensor_at(i, 14), engine);
            dnnl::memory beta2 = tensor_adapter.AsDnnlMemory(tensor_at(i, 15), engine);

            layers[i]->setWeights(queryW, queryB,
                                  keyW, keyB,
                                  valueW, valueB,
                                  att_dense_w, att_dense_b,
                                  gamma1, beta1,
                                  intermediateW, intermediateB,
                                  outputW, outputB,
                                  gamma2, beta2,
                                  bert_layers_minmax.at(i));
        }
    }

    tensorflow::Status setInputMask(const Tensor& mask)
    {
        try
        {
            switch(mask.dtype()) {
            case DT_FLOAT:
                setInputMaskWithType<float>(mask);
                break;
            case DT_DOUBLE:
                setInputMaskWithType<double>(mask);
                break;
            case DT_INT32:
                setInputMaskWithType<int32_t>(mask);
                break;
            case DT_INT64:
                setInputMaskWithType<int64_t>(mask);
                break;
            default:
                throw std::runtime_error("Unsupported mask data type");
            }
        }
        catch(const std::invalid_argument& e)
        {
            return errors::InvalidArgument(std::string{"Input mask argument error: "} + e.what());
        }
        catch(const std::out_of_range& e)
        {
            return errors::InvalidArgument(std::string{"Input mask out of range error: "} + e.what());
        }
        catch(const std::runtime_error& e)
        {
            return errors::InvalidArgument(std::string{"Input mask runtime error: "} + e.what());
        }
        return Status::OK();
    }

    template <typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
    void setInputMaskWithType(const Tensor& mask) {
        using dim = dnnl::memory::dim;

        T* data = tensor_adapter.GetValidatedTensor<T>(mask, TensorAdapter::TensorType::Mask);
        
        auto count = mask.NumElements();
        auto step = mask.dim_size(1) * mask.dim_size(2);
        auto batches = mask.dim_size(0);
        for(dim i = 0; i < batches; ++i)
        {
            ctx->setInputMask(data, count, i);

            // Advance the data pointer to the next batch element.
            // dim_size(1) can be 1 or maxTokenSize, this logic
            // should work for both.
            data += step;
            count -= step;
        }
    }

    static constexpr int max_token_size = 128;

    BertContextBuilder bert_ctx_builder;
    std::shared_ptr<BertContextT> ctx;

    std::vector<tensorflow::Tensor> bert_layer_tensors; // Used to ensure lifetime of weight tensors.
    std::vector<std::unique_ptr<BertLayer<BertContextT>>> bert_layers;
    TensorAdapter tensor_adapter;
    bool initialized;

    std::vector<Layer_minmax> bert_layers_minmax = {
        {{-10.85244083404541015625, 4.14164829254150390625}, {-1.6212508678436279296875, 2.18305110931396484375}, {-64.5349578857421875, 9.17784881591796875}, {-0.16926576197147369384765625, 12.69039154052734375}},
        {{-10.01922702789306640625, 3.2598330974578857421875}, {-2.52011966705322265625, 3.17220592498779296875}, {-70.322662353515625, 4.564808368682861328125}, {-0.16925294697284698486328125, 10.93472957611083984375}},
        {{-11.37454319000244140625, 4.04611110687255859375}, {-2.5044767856597900390625, 3.4310567378997802734375}, {-56.21540069580078125, 5.208764553070068359375}, {-0.16948534548282623291015625, 72.20577239990234375}},
        {{-14.79791736602783203125, 4.259090423583984375}, {-2.8403589725494384765625, 3.91925144195556640625}, {-93.42563629150390625, 5.099577426910400390625}, {-0.1689991652965545654296875, 9.5706195831298828125}},
        {{-13.21285343170166015625, 4.449753284454345703125}, {-3.1772515773773193359375, 4.3330135345458984375}, {-101.334869384765625, 5.41256046295166015625}, {-0.16838109493255615234375, 10.64498996734619140625}},
        {{-13.93945217132568359375, 5.1448192596435546875}, {-2.5481836795806884765625, 3.48368167877197265625}, {-91.05278778076171875, 5.9057769775390625}, {-0.16948328912258148193359375, 12.6811923980712890625}},
        {{-14.12649059295654296875, 5.23845577239990234375}, {-2.814735889434814453125, 3.2215893268585205078125}, {-89.623870849609375, 6.68107700347900390625}, {-0.16898013651371002197265625, 11.01731777191162109375}},
        {{-13.5746974945068359375, 4.71494960784912109375}, {-2.7004568576812744140625, 3.2631299495697021484375}, {-87.90279388427734375, 7.388260364532470703125}, {-0.16951541602611541748046875, 8.03197765350341796875}},
        {{-15.597011566162109375, 6.920653820037841796875}, {-3.0222375392913818359375, 3.777666568756103515625}, {-83.6142730712890625, 10.2494525909423828125}, {-0.1686449944972991943359375, 23.9402790069580078125}},
        {{-15.88373565673828125, 10.81757640838623046875}, {-2.6777179241180419921875, 3.3885133266448974609375}, {-48.061458587646484375, 16.7345333099365234375}, {-0.156786620616912841796875, 92.52396392822265625}},
        {{-18.6183719635009765625, 11.54715251922607421875}, {-2.11896610260009765625, 3.066336154937744140625}, {-41.8497314453125, 19.4496479034423828125}, {-0.16698478162288665771484375, 141.4157867431640625}},
        {{-23.8061676025390625, 11.55181217193603515625}, {-2.552584171295166015625, 3.7034885883331298828125}, {-36.45532989501953125, 16.997623443603515625}, {-0.16963402926921844482421875, 8.112117767333984375}},
    };
};


// More readable if we can somehow get this to work on TF 1.15.4

// #define REGISTER_BERT_OP(UseQuantization, UseBFloat16) \
// REGISTER_KERNEL_BUILDER(Name("Bert") \
//                             .Device(DEVICE_CPU) \
//                             .AttrConstraint("UseQuantization", UseQuantization) \
//                             .AttrConstraint("UseBFloat16", UseBFloat16), \
//                         BertOp<UseQuantization, UseBFloat16>);
    
// REGISTER_BERT_OP(false, false);
// REGISTER_BERT_OP(true, false);
// REGISTER_BERT_OP(true, true);
// REGISTER_BERT_OP(false, true);

// Fallback to type constraints to work on both TF 1.15 and 2.X

REGISTER_KERNEL_BUILDER(Name("Bert") \
                            .Device(DEVICE_CPU) \
                            .TypeConstraint("QuantizableDataType", DT_FLOAT) \
                            .TypeConstraint("NonQuantizableDataType", DT_FLOAT), \
                        BertOp<false, false>);

REGISTER_KERNEL_BUILDER(Name("Bert") \
                            .Device(DEVICE_CPU) \
                            .TypeConstraint("QuantizableDataType", DT_QINT8) \
                            .TypeConstraint("NonQuantizableDataType", DT_FLOAT), \
                        BertOp<true, false>);

REGISTER_KERNEL_BUILDER(Name("Bert") \
                            .Device(DEVICE_CPU) \
                            .TypeConstraint("QuantizableDataType", DT_FLOAT) \
                            .TypeConstraint("NonQuantizableDataType", DT_BFLOAT16), \
                        BertOp<false, true>);

REGISTER_KERNEL_BUILDER(Name("Bert") \
                            .Device(DEVICE_CPU) \
                            .TypeConstraint("QuantizableDataType", DT_QINT8) \
                            .TypeConstraint("NonQuantizableDataType", DT_BFLOAT16), \
                        BertOp<true, true>);

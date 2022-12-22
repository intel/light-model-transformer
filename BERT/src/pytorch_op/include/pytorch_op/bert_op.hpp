#ifndef LIBRARIES_AI_PERFORMANCE_MODELS_BERT_TENSOR_VALIDATOR_H
#define LIBRARIES_AI_PERFORMANCE_MODELS_BERT_TENSOR_VALIDATOR_H

#include <torch/torch.h>

#include <oneapi/dnnl/dnnl.hpp>

#include <vector>
#include <memory>

class BertLayer;
class BertContext;

namespace bert_op
{

class BertOp : public torch::CustomClassHolder {

public:
    void Configure(int64_t max_seq_len, int64_t hidden_size, int64_t intermediate_size, int64_t batch_size,
                   int64_t num_layers, bool use_quantization, bool use_bfloat16, bool calibrate_quant_factors);

    std::vector<double> GetQuantizationFactors() const;

    void Initialize(const std::vector<torch::Tensor>&, const std::vector<double>&);


    torch::Tensor Forward(torch::Tensor, torch::Tensor);

private:
    void StoreParameters(const std::vector<torch::Tensor>&);

    std::vector<dnnl::memory> parameters_;

    std::shared_ptr<BertContext> context_;
    std::vector<std::unique_ptr<BertLayer>> layers_;

};

} // namespace bert_op

#endif

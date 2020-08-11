#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "bert_layer_mb1_dynamic_tokens.h"
#include "bert_layer_batch.h"

using namespace tensorflow;

REGISTER_OP("Bert")
    .Attr("T: {int32, int64}")
    .Input("embedded: float")
    .Input("init_input: T")
    .Input("weight0_0: float")
    .Input("weight0_1: float")
    .Input("weight0_2: float")
    .Input("weight0_3: float")
    .Input("weight0_4: float")
    .Input("weight0_5: float")
    .Input("weight0_6: float")
    .Input("weight0_7: float")
    .Input("weight0_8: float")
    .Input("weight0_9: float")
    .Input("weight0_10: float")
    .Input("weight0_11: float")
    .Input("weight0_12: float")
    .Input("weight0_13: float")
    .Input("weight0_14: float")
    .Input("weight0_15: float")
    .Input("weight1_0: float")
    .Input("weight1_1: float")
    .Input("weight1_2: float")
    .Input("weight1_3: float")
    .Input("weight1_4: float")
    .Input("weight1_5: float")
    .Input("weight1_6: float")
    .Input("weight1_7: float")
    .Input("weight1_8: float")
    .Input("weight1_9: float")
    .Input("weight1_10: float")
    .Input("weight1_11: float")
    .Input("weight1_12: float")
    .Input("weight1_13: float")
    .Input("weight1_14: float")
    .Input("weight1_15: float")
    .Input("weight2_0: float")
    .Input("weight2_1: float")
    .Input("weight2_2: float")
    .Input("weight2_3: float")
    .Input("weight2_4: float")
    .Input("weight2_5: float")
    .Input("weight2_6: float")
    .Input("weight2_7: float")
    .Input("weight2_8: float")
    .Input("weight2_9: float")
    .Input("weight2_10: float")
    .Input("weight2_11: float")
    .Input("weight2_12: float")
    .Input("weight2_13: float")
    .Input("weight2_14: float")
    .Input("weight2_15: float")
    .Input("weight3_0: float")
    .Input("weight3_1: float")
    .Input("weight3_2: float")
    .Input("weight3_3: float")
    .Input("weight3_4: float")
    .Input("weight3_5: float")
    .Input("weight3_6: float")
    .Input("weight3_7: float")
    .Input("weight3_8: float")
    .Input("weight3_9: float")
    .Input("weight3_10: float")
    .Input("weight3_11: float")
    .Input("weight3_12: float")
    .Input("weight3_13: float")
    .Input("weight3_14: float")
    .Input("weight3_15: float")
    .Input("weight4_0: float")
    .Input("weight4_1: float")
    .Input("weight4_2: float")
    .Input("weight4_3: float")
    .Input("weight4_4: float")
    .Input("weight4_5: float")
    .Input("weight4_6: float")
    .Input("weight4_7: float")
    .Input("weight4_8: float")
    .Input("weight4_9: float")
    .Input("weight4_10: float")
    .Input("weight4_11: float")
    .Input("weight4_12: float")
    .Input("weight4_13: float")
    .Input("weight4_14: float")
    .Input("weight4_15: float")
    .Input("weight5_0: float")
    .Input("weight5_1: float")
    .Input("weight5_2: float")
    .Input("weight5_3: float")
    .Input("weight5_4: float")
    .Input("weight5_5: float")
    .Input("weight5_6: float")
    .Input("weight5_7: float")
    .Input("weight5_8: float")
    .Input("weight5_9: float")
    .Input("weight5_10: float")
    .Input("weight5_11: float")
    .Input("weight5_12: float")
    .Input("weight5_13: float")
    .Input("weight5_14: float")
    .Input("weight5_15: float")
    .Input("weight6_0: float")
    .Input("weight6_1: float")
    .Input("weight6_2: float")
    .Input("weight6_3: float")
    .Input("weight6_4: float")
    .Input("weight6_5: float")
    .Input("weight6_6: float")
    .Input("weight6_7: float")
    .Input("weight6_8: float")
    .Input("weight6_9: float")
    .Input("weight6_10: float")
    .Input("weight6_11: float")
    .Input("weight6_12: float")
    .Input("weight6_13: float")
    .Input("weight6_14: float")
    .Input("weight6_15: float")
    .Input("weight7_0: float")
    .Input("weight7_1: float")
    .Input("weight7_2: float")
    .Input("weight7_3: float")
    .Input("weight7_4: float")
    .Input("weight7_5: float")
    .Input("weight7_6: float")
    .Input("weight7_7: float")
    .Input("weight7_8: float")
    .Input("weight7_9: float")
    .Input("weight7_10: float")
    .Input("weight7_11: float")
    .Input("weight7_12: float")
    .Input("weight7_13: float")
    .Input("weight7_14: float")
    .Input("weight7_15: float")
    .Input("weight8_0: float")
    .Input("weight8_1: float")
    .Input("weight8_2: float")
    .Input("weight8_3: float")
    .Input("weight8_4: float")
    .Input("weight8_5: float")
    .Input("weight8_6: float")
    .Input("weight8_7: float")
    .Input("weight8_8: float")
    .Input("weight8_9: float")
    .Input("weight8_10: float")
    .Input("weight8_11: float")
    .Input("weight8_12: float")
    .Input("weight8_13: float")
    .Input("weight8_14: float")
    .Input("weight8_15: float")
    .Input("weight9_0: float")
    .Input("weight9_1: float")
    .Input("weight9_2: float")
    .Input("weight9_3: float")
    .Input("weight9_4: float")
    .Input("weight9_5: float")
    .Input("weight9_6: float")
    .Input("weight9_7: float")
    .Input("weight9_8: float")
    .Input("weight9_9: float")
    .Input("weight9_10: float")
    .Input("weight9_11: float")
    .Input("weight9_12: float")
    .Input("weight9_13: float")
    .Input("weight9_14: float")
    .Input("weight9_15: float")
    .Input("weight10_0: float")
    .Input("weight10_1: float")
    .Input("weight10_2: float")
    .Input("weight10_3: float")
    .Input("weight10_4: float")
    .Input("weight10_5: float")
    .Input("weight10_6: float")
    .Input("weight10_7: float")
    .Input("weight10_8: float")
    .Input("weight10_9: float")
    .Input("weight10_10: float")
    .Input("weight10_11: float")
    .Input("weight10_12: float")
    .Input("weight10_13: float")
    .Input("weight10_14: float")
    .Input("weight10_15: float")
    .Input("weight11_0: float")
    .Input("weight11_1: float")
    .Input("weight11_2: float")
    .Input("weight11_3: float")
    .Input("weight11_4: float")
    .Input("weight11_5: float")
    .Input("weight11_6: float")
    .Input("weight11_7: float")
    .Input("weight11_8: float")
    .Input("weight11_9: float")
    .Input("weight11_10: float")
    .Input("weight11_11: float")
    .Input("weight11_12: float")
    .Input("weight11_13: float")
    .Input("weight11_14: float")
    .Input("weight11_15: float")
    .Output("berted: float");

#define LAYERS 12
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

template<typename T>
class BertOp : public OpKernel {
public:
  explicit BertOp(OpKernelConstruction* context) : OpKernel(context) {
      printf("Bert op Construction!\n");
      //setenv("KMP_AFFINITY", "granularity=fine,compact,1,0", 1);
      //setenv("KMP_BLOCKTIME", "1", 1);
  
      for (int i = 0; i < LAYERS; ++i) {
          bert_layers[i] = NULL;
      }
      for (int i = 0; i < LAYERS; ++i) {
          batch_bert_layers[i] = NULL;
      }
      
      initialized = false;
      batch_mode = false;
  }

  ~BertOp() {
      for (int i = 0; i < LAYERS; ++i) {
          delete bert_layers[i];
      }
      for (int i = 0; i < LAYERS; ++i) {
          delete batch_bert_layers[i];
      }
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& tensor_embeded = context->input(0);
    const Tensor& tensor_ids = context->input(1);
    float *embeded = (float *)tensor_embeded.tensor_data().data();
    T *ids = (T *)tensor_ids.tensor_data().data();

    int batch_size = tensor_ids.dim_size(0);
    int total_tokens = tensor_embeded.dim_size(0); // total_tokens = batch_size * tokens_each 
    int input_tokens = total_tokens / batch_size;
    int ids_len = tensor_ids.dim_size(1);

    // Initialize the weights and mode
    if (!initialized) {
        this->batch_mode = (batch_size > 1);
        if (this->batch_mode) {
            for (int i = 0; i < LAYERS; ++i) {
                batch_bert_layers[i] = new BatchBertLayer<maxTokenSize, hiddenSize, intermediateSize, attentionHeadNum>(i);
            }
        } else {
            for (int i = 0; i < LAYERS; ++i) {
                bert_layers[i] = new BertLayer(i, maxTokenSize);
            }
        }
        initWeights(context);
        initialized = true;
    }

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, tensor_embeded.shape(),
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();
   
    hpj::Matrix<float> input_buf(embeded, total_tokens, hiddenSize, hiddenSize);
    hpj::Matrix<float> *m_data = &input_buf;
    if (!this->batch_mode) {
        if (unlikely(batch_size > 1)) {
            printf("ERROR: initialized with mini_batch=1, but now mini_batch=%d\n", batch_size);
            exit(-1);
        }
        // Get input mask (ends with how many zeros)
        int zeros = 0;
        for (int i = input_tokens-1; i >= 0; --i) {
            if (ids[i] == 0) { zeros += 1; }
            else { break; }
        }
        int actual_tokens = ids_len - zeros;
        for (int i = 0; i < LAYERS; ++i) {
            hpj::Matrix<float> &out = bert_layers[i]->forward(*m_data, input_tokens, actual_tokens);
            m_data = &out;
        }
    } else {
        std::vector<int> actual_tokens;
        for (int b = 0; b < batch_size; ++b) {
            int zeros = 0;
            for (int i = ids_len-1; i >= 0; --i) {
                if (ids[i] == 0) { zeros += 1; }
                else { break; }
            }
            actual_tokens.push_back(ids_len - zeros);
            ids += ids_len;
        }
        for (int i = 0; i < LAYERS; ++i) {
            hpj::Matrix<float> &out = batch_bert_layers[i]->forward(*m_data, actual_tokens);
            m_data = &out;
        }
    }

    // Copy data to output
    #pragma omp parallel for
    for (int i = 0; i < total_tokens; ++i) {
      memcpy(output + i * hiddenSize, m_data->Row(i), sizeof(float) * hiddenSize);
    }
  }

private:
  void initWeights(OpKernelContext* context) {
    int idx = 2;
    for (int i = 0; i < LAYERS; ++i) {
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

        if (this->batch_mode)   {
            batch_bert_layers[i]->setWeights(queryW, queryB,
                               keyW, keyB,
                               valueW, valueB,
                               att_dense_w, att_dense_b, 
                               gamma1, beta1,
                               intermediateW, intermediateB,
                               outputW, outputB,
                               gamma2, beta2);
        } else {
            bert_layers[i]->setWeights(queryW, queryB,
                               keyW, keyB,
                               valueW, valueB,
                               att_dense_w, att_dense_b, 
                               gamma1, beta1,
                               intermediateW, intermediateB,
                               outputW, outputB,
                               gamma2, beta2);
        } // end else
    }
  }

  // At the beginning, we load weights from file. It can make this op have only 2 inputs
  // It could be even faster and consume less memory. 
  // However, it makes the pb file not self-contained any more
  void readValueFromFile(const char *filename, float *pvalue, int size) {
    FILE *fp = fopen(filename, "rb");
    if (fp) {
      int ret = fread(pvalue, 4, size, fp);
      fclose(fp);
      if (ret != size) {
        printf("Cannot read %d float values from %s\n", size, filename);
        exit(-1);
      }
    } else {
      printf("Cannot open weight file: %s\n", filename);
      exit(-1);
    }
  }

private:
  static const int maxTokenSize = 128;
  static const int hiddenSize = 768;
  static const int intermediateSize = 3072;
  static const int attentionHeadNum = 12;

  BertLayer *bert_layers[LAYERS];
  BatchBertLayer<maxTokenSize, hiddenSize, intermediateSize, attentionHeadNum> *batch_bert_layers[LAYERS];

  bool initialized;
  bool batch_mode;
};


REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<int32>("T"), BertOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Bert").Device(DEVICE_CPU).TypeConstraint<int64>("T"), BertOp<int64>);

#ifndef LIBRARIES_AI_PERFORMANCE_MODELS_BERT_BERT_TYPE_TRAITS_H
#define LIBRARIES_AI_PERFORMANCE_MODELS_BERT_BERT_TYPE_TRAITS_H

#include "dnnl_common.h"

#include <type_traits>

template <template <class...> class, class...>
struct is_template_instance : public std::false_type {};

template <template <class...> class T, class... U>
struct is_template_instance<T, T<U...>> : public std::true_type {};

template <bool B>
struct use_quantization : public std::conditional<B, int8_t, float> {};

template <bool B>
struct use_bfloat16 : public std::conditional<B, ::bfloat16, float> {};

#endif

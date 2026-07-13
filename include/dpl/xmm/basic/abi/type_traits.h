// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/basic/abi.h"

#include "dpl/xmm/basic/abi/xmmdef.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/core/concepts/same_as.h"
#    include "dpl/core/type_traits/remove_cv.h"
#    include "dpl/core/type_traits/simd_element_representation.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <typename T>
using representation DPL_NODEBUG = dx::simd_element_representation<abi_tag, T>;

template <typename T>
using representation_t DPL_NODEBUG =
    dx::simd_element_representation_t<abi_tag, T>;

template <typename T>
struct native_vector {};

template <typename T>
using native_vector_t DPL_NODEBUG =
    typename native_vector<representation_t<T>>::type;

template <>
struct native_vector<float> {
    using type DPL_NODEBUG = __m128;
};
template <>
struct native_vector<double> {
    using type DPL_NODEBUG = __m128d;
};
template <>
struct native_vector<ext::bfloat16> {
    using type DPL_NODEBUG = __m128bh;
};
template <>
struct native_vector<ext::float16> {
    using type DPL_NODEBUG = __m128h;
};

template <integral T>
requires different_from<remove_cv_t<T>, bool>
struct native_vector<T> {
    using type DPL_NODEBUG = __m128i;
};
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/basic/abi.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi/type_traits.h"
#  include "dpl/xmm/basic/abi/xmmdef.h"

#  if !DPL_MODULES
#    include "dpl/core/concepts/simd_element.h"
#    include "dpl/core/type_traits/simd_element_representation.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <typename T>
concept native_vector_type = same_as<T, __m128> //
    || same_as<T, __m128d>                      //
    || same_as<T, __m128i>                      //
    || same_as<T, __m128bh>                     //
    || same_as<T, __m128h>;

template <typename T>
concept vectorizable = requires { typename native_vector_t<T>; };

template <typename E>
concept simd_element = simd_element_for<E, abi_tag>;

template <typename E, size_t N>
concept sized_element = sizeof(E) == N && simd_element<E>;

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

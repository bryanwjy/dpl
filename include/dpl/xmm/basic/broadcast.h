// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/constants/all_bits.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/utility/sequence.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(abi_tag tag, type_identity_t<E> scalar) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
        constexpr auto forward = [](auto, E scalar) { return scalar; };
        return [forward]<size_t... Is>(
                   index_sequence<Is...>, abi_tag tag, E data) {
            return dx::xmm::initialize<E>(tag, forward(Is, data)...);
        }(iota_sequence<E, abi_tag>, tag, scalar);
    } else {
        if constexpr (common_float_with<float, E>) {
            return _mm_set1_ps(scalar);
        } else if constexpr (common_float_with<double, E>) {
            return _mm_set1_pd(scalar);
        } else if constexpr (bfloat16_like<E> || float16_like<E>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_set1_epi16(__DPL bit_cast<int16>(scalar)));
        } else if constexpr (common_bits_with<E, int32>) {
            return _mm_set1_epi32(__DPL bit_cast<int32>(scalar));
        } else if constexpr (common_bits_with<E, int16>) {
            return _mm_set1_epi16(__DPL bit_cast<int16>(scalar));
        } else if constexpr (common_bits_with<E, int8>) {
            return _mm_set1_epi8(__DPL bit_cast<int8>(scalar));
        } else {
            static_assert(common_bits_with<E, int64>);
            return _mm_set1_epi64x(__DPL bit_cast<int64>(scalar));
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(abi_tag tag, same_as<bool> auto scalar) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    return +xmm::broadcast(tag, scalar ? dx::all_bits_v<E> : dx::zero);
}
} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

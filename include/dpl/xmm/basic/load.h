// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/basic/aligned.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/utility/sequence.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr simd<E> load(abi_tag tag, E const* data) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
        return []<size_t... Is>(
                   index_sequence<Is...>, abi_tag tag, E const* data) {
            return dx::xmm::initialize<E>(tag, data[Is]...);
        }(iota<E>, tag, data);
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_loadu_ps(reinterpret_cast<float const*>(data));
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_loadu_pd(reinterpret_cast<double const*>(data));
        } else if constexpr (bfloat16_like<representation_t<E>>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(data)));
        } else if constexpr (float16_like<representation_t<E>>) {
#if DPL_SIMD_X86_AVX512FP16
            return _mm_castsi128_ph(
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(data)));
#else
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(data)));
#endif
        } else {
            return _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr simd<E> load(abi_tag tag, aligned_t, E const* data) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
        return xmm::load(tag, data);
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_load_ps(reinterpret_cast<float const*>(data));
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_load_pd(reinterpret_cast<double const*>(data));
        } else if constexpr (bfloat16_like<representation_t<E>>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_load_si128(reinterpret_cast<__m128i const*>(data)));
        } else if constexpr (float16_like<representation_t<E>>) {
            return _mm_castsi128_ph(
                _mm_load_si128(reinterpret_cast<__m128i const*>(data)));
        } else {
            return _mm_load_si128(reinterpret_cast<__m128i const*>(data));
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr simd<E> load(E const* data) noexcept {
    return xmm::load(xmm::abi, data);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr simd<E> aligned_load(E const* data) noexcept {
    return xmm::load(xmm::abi, dx::aligned, data);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

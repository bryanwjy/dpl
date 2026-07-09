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
#  include "dpl/std/type_traits/sequence.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E> load(E const* data) noexcept {
    if consteval {
        return []<size_t... Is>(index_sequence<Is...>, E const* data) {
            return dx::xmm::initialize<E>(data[Is]...);
        }(iota<E>, data);
    } else {
        if constexpr (is_same_v<float, E>) {
            return _mm_loadu_ps(reinterpret_cast<float const*>(data));
        } else if constexpr (is_same_v<double, E>) {
            return _mm_loadu_pd(reinterpret_cast<double const*>(data));
        } else if constexpr (is_same_v<ext::bfloat16, E>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(data)));
        } else if constexpr (is_same_v<ext::float16, E>) {
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
constexpr vector<E> load(aligned_t, E const* data) noexcept {
    if consteval {
        return xmm::load(data);
    } else {
        if constexpr (is_same_v<float, E>) {
            return _mm_load_ps(reinterpret_cast<float const*>(data));
        } else if constexpr (is_same_v<double, E>) {
            return _mm_load_pd(reinterpret_cast<double const*>(data));
        } else if constexpr (is_same_v<ext::bfloat16, E>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_load_si128(reinterpret_cast<__m128i const*>(data)));
        } else if constexpr (is_same_v<ext::float16, E>) {
            return _mm_castsi128_ph(
                _mm_load_si128(reinterpret_cast<__m128i const*>(data)));
        } else {
            return _mm_load_si128(reinterpret_cast<__m128i const*>(data));
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E> load(abi_tag, E const* data) noexcept
requires requires { xmm::load(data); }
{
    return xmm::load(data);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E> load(abi_tag, aligned_t aligned, E const* data) noexcept
requires requires { xmm::load(aligned, data); }
{
    return xmm::load(aligned, data);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/extract.h"

#if !DPL_MODULES
#  include "dpl/core/basic/aligned.h"
#  include "dpl/std/bit/bit_cast.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element E>
__DPL_HIDE_FROM_ABI constexpr void store(simd<E> src, E* dst) noexcept {
    if consteval {
        for (auto i = 0zu; i < src.size(); ++i) {
            dst[i] = xmm::extract(src, i);
        }
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            _mm_storeu_ps(reinterpret_cast<float*>(dst), +src);
        } else if constexpr (same_as<double, representation_t<E>>) {
            _mm_storeu_pd(reinterpret_cast<double*>(dst), +src);
        } else if constexpr (integral<representation_t<E>>) {
            static_assert(integral<representation_t<E>>);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), +src);
        } else if constexpr (bfloat16_like<representation_t<E>>) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                __DPL bit_cast<__m128i>(+src));
        } else {
            static_assert(same_as<__m128h, native_vector_t<E>>);
#if DPL_SIMD_X86_AVX512FP16
            _mm_storeu_ph(dst, +src);
#else
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                __DPL bit_cast<__m128i>(+src));
#endif
        }
    }
}

DPL_EXPORT template <simd_element E>
__DPL_HIDE_FROM_ABI constexpr void store(
    aligned_t, simd<E> src, E* dst) noexcept {
    if consteval {
        return xmm::store(src, dst);
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            _mm_store_ps(reinterpret_cast<float*>(dst), +src);
        } else if constexpr (same_as<double, representation_t<E>>) {
            _mm_store_pd(reinterpret_cast<double*>(dst), +src);
        } else if constexpr (integral<representation_t<E>>) {
            static_assert(integral<E>);
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), +src);
        } else if constexpr (bfloat16_like<representation_t<E>>) {
            _mm_store_si128(reinterpret_cast<__m128i*>(dst),
                __DPL bit_cast<__m128i>(+src));
        } else {
            static_assert(same_as<__m128h, native_vector_t<E>>);
#if DPL_SIMD_X86_AVX512FP16
            _mm_store_ph(dst, +src);
#else
            _mm_store_si128(reinterpret_cast<__m128i*>(dst),
                __DPL bit_cast<__m128i>(+src));
#endif
        }
    }
}

DPL_EXPORT template <simd_element E>
__DPL_HIDE_FROM_ABI constexpr void store(abi_tag, simd<E> src, E* dst) noexcept
requires requires { xmm::store(src, dst); }
{
    xmm::store(src, dst);
}

DPL_EXPORT template <simd_element E>
__DPL_HIDE_FROM_ABI constexpr void store(
    abi_tag, aligned_t aligned, simd<E> src, E* dst) noexcept
requires requires { xmm::store(aligned, src, dst); }
{
    xmm::store(aligned, src, dst);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

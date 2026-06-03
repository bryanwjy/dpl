// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/extract.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_element_for.h"
#  include "dpl/std/bit/bit_cast.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element_for<abi_tag> E>
__DPL_HIDE_FROM_ABI constexpr void store(
    abi_tag tag, simd<E> src, E* dst) noexcept {
    if consteval {
        return []<size_t I = 0>(this auto self, abi_tag tag, simd<E> src,
                   E* dst, immediate<I> idx = {}) {
            if constexpr (I == simd<E>::size()) {
                return;
            } else {
                dst[I] = dx::xmm::extract(tag, src, idx);
                self(tag, src, dst, dx::imm<I + 1>);
            }
        }(tag, src, dst);
    } else {
        if constexpr (same_as<float, lane_representation_t<E>>) {
            _mm_storeu_ps(reinterpret_cast<float*>(dst), +src);
        } else if constexpr (same_as<double, lane_representation_t<E>>) {
            _mm_storeu_pd(reinterpret_cast<double*>(dst), +src);
        } else if constexpr (integral<lane_representation_t<E>>) {
            static_assert(integral<lane_representation_t<E>>);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), +src);
        } else if constexpr (bfloat16_like<lane_representation_t<E>>) {
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

DPL_EXPORT template <simd_element_for<abi_tag> E>
__DPL_HIDE_FROM_ABI constexpr void aligned_store(
    abi_tag tag, simd<E> src, E* dst) noexcept {
    if consteval {
        return dx::xmm::store(tag, src, dst);
    } else {
        if constexpr (same_as<float, lane_representation_t<E>>) {
            _mm_store_ps(reinterpret_cast<float*>(dst), +src);
        } else if constexpr (same_as<double, lane_representation_t<E>>) {
            _mm_store_pd(reinterpret_cast<double*>(dst), +src);
        } else if constexpr (integral<lane_representation_t<E>>) {
            static_assert(integral<E>);
            _mm_store_si128(reinterpret_cast<__m128i*>(dst), +src);
        } else if constexpr (bfloat16_like<lane_representation_t<E>>) {
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

DPL_EXPORT template <simd_element_for<abi_tag> E>
__DPL_HIDE_FROM_ABI constexpr void store(simd<E> src, E* dst) noexcept {
    xmm::store(xmm::abi, src, dst);
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
__DPL_HIDE_FROM_ABI constexpr void aligned_store(simd<E> src, E* dst) noexcept {
    xmm::aligned_store(xmm::abi, src, dst);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/load.h"
#    include "dpl/xmm/basic/store.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> reinterpret(vector<F> src) noexcept {
    if consteval {
        if constexpr (is_same_v<native_vector_t<E>, native_vector_t<F>>) {
            return +src;
        } else {
            // Need to work around unions (no consteval bitcast) in MSVC
            struct alignas(xmm::abi_tag::alignment) {
                F data[xmm::abi_tag::size / sizeof(F)];
            } tmp{};
            xmm::store(src, tmp.data);
            struct alignas(xmm::abi_tag::alignment) dst_t {
                E data[xmm::abi_tag::size / sizeof(E)];
            } dst = __DPL bit_cast<dst_t>(tmp);
            return xmm::load<E>(dst.data);
        }
    } else {
        if constexpr (is_same_v<E, float>) {
            if constexpr (integral<F>) {
                return _mm_castsi128_ps(+src);
            } else if constexpr (is_same_v<F, double>) {
                return _mm_castpd_ps(+src);
            } else if constexpr (is_same_v<F, float>) {
                return +src;
            } else if constexpr (is_same_v<F, ext::bfloat16>) {
                static_assert(is_same_v<__m128bh, native_vector_t<F>>);
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(is_same_v<F, ext::float16>);
#  if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_ps(+src);
#  else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#  endif
            }
        } else if constexpr (is_same_v<E, double>) {
            if constexpr (is_same_v<F, float>) {
                return _mm_castps_pd(+src);
            } else if constexpr (integral<F>) {
                return _mm_castsi128_pd(+src);
            } else if constexpr (is_same_v<F, double>) {
                return +src;
            } else if constexpr (is_same_v<F, ext::bfloat16>) {
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(is_same_v<F, ext::float16>);
#  if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_pd(+src);
#  else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#  endif
            }
        } else if constexpr (integral<E>) {
            if constexpr (is_same_v<F, double>) {
                return _mm_castpd_si128(+src);
            } else if constexpr (is_same_v<F, float>) {
                return _mm_castps_si128(+src);
            } else if constexpr (integral<F>) {
                return +src;
            } else if constexpr (is_same_v<F, ext::bfloat16>) {
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(is_same_v<F, ext::float16>);
#  if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_si128(+src);
#  else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#  endif
            }
        } else if constexpr (is_same_v<E, ext::bfloat16>) {
            return __DPL bit_cast<__m128bh>(+src);
        } else {
            static_assert(is_same_v<E, ext::float16>);
#  if DPL_SIMD_X86_AVX512FP16
            if constexpr (is_same_v<F, double>) {
                return _mm_castpd_ph(+src);
            } else if constexpr (is_same_v<F, float>) {
                return _mm_castps_ph(+src);
            } else if constexpr (is_same_v<F, ext::bfloat16>) {
                return __DPL bit_cast<__m128bh>(+src);
            } else if constexpr (integral<F>) {
                return _mm_castsi128_ph(+src);
            } else {
                return +src;
            }
#  else
            return __DPL bit_cast<native_vector_t<E>>(+src);
#  endif
        }
    }
}

template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> reinterpret(mask<F> src) noexcept {
    return +xmm::reinterpret<E>(vector<F>(+src));
}

template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> reinterpret(abi_tag, vector<F> src) noexcept {
    return xmm::reinterpret<E>(src);
}

template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr mask<E> reinterpret(abi_tag, mask<F> src) noexcept {
    return xmm::reinterpret<E>(src);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

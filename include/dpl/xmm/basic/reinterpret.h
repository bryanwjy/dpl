// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/load.h"
#include "dpl/xmm/basic/store.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/bit/bit_cast.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> reinterpret(abi_tag tag, simd<F> src) noexcept {
    if consteval {
        // Need to work around unions (no consteval bitcast) in MSVC
        struct alignas(16) {
            F data[element_count<F, xmm::abi_tag>];
        } tmp{};
        xmm::store(tag, src, tmp.data);
        struct alignas(16) dst_t {
            E data[element_count<E, xmm::abi_tag>];
        } dst = __DPL bit_cast<dst_t>(tmp);
        return xmm::load<E>(tag, dst.data);
    } else {
        if constexpr (common_float_with<E, float>) {
            if constexpr (same_as<__m128i, native_vector_t<F>>) {
                return _mm_castsi128_ps(+src);
            } else if constexpr (same_as<__m128d, native_vector_t<F>>) {
                return _mm_castpd_ps(+src);
            } else if constexpr (same_as<__m128, native_vector_t<F>>) {
                return +src;
            } else if constexpr (brain_float<F>) {
                static_assert(same_as<__m128bh, native_vector_t<F>>);
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(same_as<__m128h, native_vector_t<F>>);
#if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_ps(+src);
#else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
            }
        } else if constexpr (common_float_with<E, double>) {
            if constexpr (common_float_with<float, F>) {
                return _mm_castps_pd(+src);
            } else if constexpr (same_as<__m128i, native_vector_t<F>>) {
                return _mm_castsi128_pd(+src);
            } else if constexpr (same_as<__m128d, native_vector_t<F>>) {
                return +src;
            } else if constexpr (brain_float<F>) {
                static_assert(same_as<__m128bh, native_vector_t<F>>);
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(same_as<__m128h, native_vector_t<F>>);
#if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_pd(+src);
#else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
            }
        } else if constexpr (integral<E> || enumeration<E>) {
            if constexpr (same_as<__m128d, native_vector_t<F>>) {
                return _mm_castpd_si128(+src);
            } else if constexpr (common_float_with<float, F>) {
                return _mm_castps_si128(+src);
            } else if constexpr (same_as<__m128i, native_vector_t<F>>) {
                return +src;
            } else if constexpr (brain_float<F>) {
                static_assert(same_as<__m128bh, native_vector_t<F>>);
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(same_as<__m128h, native_vector_t<F>>);
#if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_si128(+src);
#else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
            }
        } else if constexpr (brain_float<E>) {
            static_assert(same_as<__m128bh, native_vector_t<E>>);
            return __DPL bit_cast<native_vector_t<E>>(+src);
        } else {
            static_assert(same_as<__m128h, native_vector_t<E>>);
#if DPL_SIMD_X86_AVX512FP16
            if constexpr (common_float_with<E, F>) {
                return +src;
            } else if constexpr (same_as<__m128d, native_vector_t<F>>) {
                return _mm_castpd_ph(+src);
            } else if constexpr (common_float_with<float, F>) {
                return _mm_castps_ph(+src);
            } else if constexpr (brain_float<F>) {
                static_assert(same_as<__m128bh, native_vector_t<F>>);
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(same_as<__m128i, native_vector_t<F>>);
                return _mm_castsi128_ph(+src);
            }
#else
            return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
        }
    }
}

DPL_EXPORT template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> reinterpret(abi_tag tag, mask<F> src) noexcept {
    return +dx::xmm::reinterpret<E>(tag, simd<F>(+src));
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

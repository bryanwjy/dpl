// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/load.h"
#  include "dpl/xmm/basic/store.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> reinterpret(simd<F> src) noexcept {
    if consteval {
        // Need to work around unions (no consteval bitcast) in MSVC
        struct alignas(xmm::abi_tag::alignment) {
            F data[xmm::abi_tag::size / sizeof(F)];
        } tmp{};
        xmm::store(src, tmp.data);
        struct alignas(xmm::abi_tag::alignment) dst_t {
            E data[xmm::abi_tag::size / sizeof(E)];
        } dst = __DPL bit_cast<dst_t>(tmp);
        return xmm::load<E>(dst.data);
    } else {
        if constexpr (is_same_v<representation_t<E>, float>) {
            if constexpr (is_same_v<__m128i, native_vector_t<F>>) {
                return _mm_castsi128_ps(+src);
            } else if constexpr (is_same_v<__m128d, native_vector_t<F>>) {
                return _mm_castpd_ps(+src);
            } else if constexpr (is_same_v<__m128, native_vector_t<F>>) {
                return +src;
            } else if constexpr (bfloat16_like<F>) {
                static_assert(is_same_v<__m128bh, native_vector_t<F>>);
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(is_same_v<__m128h, native_vector_t<F>>);
#if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_ps(+src);
#else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
            }
        } else if constexpr (is_same_v<representation_t<E>, double>) {
            if constexpr (is_same_v<__m128, native_vector_t<F>>) {
                return _mm_castps_pd(+src);
            } else if constexpr (is_same_v<__m128i, native_vector_t<F>>) {
                return _mm_castsi128_pd(+src);
            } else if constexpr (is_same_v<__m128d, native_vector_t<F>>) {
                return +src;
            } else if constexpr (is_same_v<__m128bh, native_vector_t<F>>) {
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(is_same_v<__m128h, native_vector_t<F>>);
#if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_pd(+src);
#else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
            }
        } else if constexpr (integral<representation_t<E>>) {
            if constexpr (is_same_v<__m128d, native_vector_t<F>>) {
                return _mm_castpd_si128(+src);
            } else if constexpr (is_same_v<__m128, native_vector_t<F>>) {
                return _mm_castps_si128(+src);
            } else if constexpr (is_same_v<__m128i, native_vector_t<F>>) {
                return +src;
            } else if constexpr (is_same_v<__m128bh, native_vector_t<F>>) {
                return __DPL bit_cast<native_vector_t<E>>(+src);
            } else {
                static_assert(is_same_v<__m128h, native_vector_t<F>>);
#if DPL_SIMD_X86_AVX512FP16
                return _mm_castph_si128(+src);
#else
                return __DPL bit_cast<native_vector_t<E>>(+src);
#endif
            }
        } else if constexpr (same_as<__m128bh, native_vector_t<E>>) {
            return __DPL bit_cast<native_vector_t<E>>(+src);
        } else {
            static_assert(same_as<__m128h, native_vector_t<E>>);
#if DPL_SIMD_X86_AVX512FP16
            if constexpr (same_as<native_vector_t<E>, native_vector_t<F>>) {
                return +src;
            } else if constexpr (same_as<__m128d, native_vector_t<F>>) {
                return _mm_castpd_ph(+src);
            } else if constexpr (same_as<__m128, native_vector_t<F>>) {
                return _mm_castps_ph(+src);
            } else if constexpr (same_as<__m128bh, native_vector_t<F>>) {
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
constexpr mask<E> reinterpret(mask<F> src) noexcept {
    return +xmm::reinterpret<E>(simd<F>(+src));
}

DPL_EXPORT template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr simd<E> reinterpret(abi_tag, simd<F> src) noexcept {
    return xmm::reinterpret<E>(src);
}

DPL_EXPORT template <simd_element E, simd_element F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr mask<E> reinterpret(abi_tag, mask<F> src) noexcept {
    return xmm::reinterpret<E>(src);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

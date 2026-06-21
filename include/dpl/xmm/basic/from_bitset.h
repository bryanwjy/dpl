// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/immediate/constants/all_bits.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/to_signed.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

namespace internal {
template <typename E>
struct alignas(16) mask_lane_bits_t {
    signed_representation_t<E> data[abi_tag::size / sizeof(E)];

    __m128i operator+() const noexcept {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(data));
    }
};
template <typename E>
inline constexpr auto mask_lane_bits = []<size_t... Is>(index_sequence<Is...>) {
    using type = signed_representation_t<E>;
    return mask_lane_bits_t<E>{static_cast<type>(1 << Is)...};
}(iota<E>);
} // namespace internal

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> from_bitset(
    bitset<simd_abi_traits<E, abi_tag>::size> mask) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    constexpr auto width = simd_abi_traits<E, abi_tag>::size;
    if consteval {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return +xmm::initialize<E>(
                abi_tag{}, (mask[Is] ? dx::all_bits_v<E> : dx::zero_v<E>)...);
        }(iota<E>);
    } else {
        auto const xmm0 = [&]() {
            if constexpr (width == 16) {
                auto const imask = __DPL to_underlying(mask);
                auto const bits = +internal::mask_lane_bits<int16>;
                auto const upper = _mm_set1_epi16(imask >> 8);
                auto const lower = _mm_set1_epi16(imask & 0xff);
                auto const zero = _mm_setzero_si128();
                auto const hi8 =
                    _mm_cmpgt_epi16(_mm_and_si128(upper, bits), zero);
                auto const lo8 =
                    _mm_cmpgt_epi16(_mm_and_si128(lower, bits), zero);
                return _mm_packs_epi16(lo8, hi8);
            } else if constexpr (width == 8) {
                auto const val =
                    static_cast<uint16>(__DPL to_underlying(mask));
                auto const m = _mm_set1_epi16( __DPL to_signed(val));
                return _mm_cmpgt_epi16(
                    _mm_and_si128(m, +internal::mask_lane_bits<E>),
                    _mm_setzero_si128());
            } else if constexpr (width == 4) {
                auto const val =
                    static_cast<uint32>(__DPL to_underlying(mask));
                auto const m = _mm_set1_epi32( __DPL to_signed(val));
                return _mm_cmpgt_epi32(
                    _mm_and_si128(m, +internal::mask_lane_bits<E>),
                    _mm_setzero_si128());
            } else if constexpr (width == 2) {
                auto const val =
                    static_cast<uint64>(__DPL to_underlying(mask));
                auto const m = _mm_set1_epi64x( __DPL to_signed(val));
                return _mm_cmpgt_epi64(
                    _mm_and_si128(m, +internal::mask_lane_bits<E>),
                    _mm_setzero_si128());
            } else {
                static_assert(width == 1);
                return _mm_set1_epi32(-( __DPL to_underlying(mask) != 0));
            }
        }();
        if constexpr (same_as<native_vector_t<E>, __m128>) {
            return _mm_castsi128_ps(xmm0);
        } else if constexpr (same_as<native_vector_t<E>, __m128d>) {
            return _mm_castsi128_pd(xmm0);
        } else if constexpr (same_as<native_vector_t<E>, __m128i>) {
            return xmm0;
        } else {
            return __DPL bit_cast<native_vector_t<E>>(xmm0);
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> from_bitset(
    abi_tag tag, bitset<simd_abi_traits<E, abi_tag>::size> mask) noexcept {
    return xmm::from_bitset<E>(mask);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif

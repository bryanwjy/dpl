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

#if !DPL_MODULES
#  include "dpl/core/constants/all_bits.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/type_traits/type_identity.h" // IWYU pragma: keep
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/forward.h"       // IWYU pragma: keep
#  include "dpl/std/utility/to_signed.h"     // IWYU pragma: keep
#  include "dpl/std/utility/to_underlying.h" // IWYU pragma: keep

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

DPL_EXPORT template <simd_element E, convertible_to<E>... Args>
requires (... && !same_as<Args, bool>) &&
    (sizeof...(Args) == simd_abi_traits<E, abi_tag>::size)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> initialize(abi_tag tag, Args&&... args) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
#if !DPL_COMPILER_MSVC
        using array = E[abi_tag::size / sizeof(E)];
        alignas(abi_tag::alignment)
            array buffer{static_cast<E>(__DPL forward<Args>(args))...};
        return __DPL bit_cast<native_vector_t<E>>(buffer);
#else
        if constexpr (same_as<float, representation_t<E>>) {
            return native_vector_t<E>{
                .m128_f32 = {__DPL bit_cast<float>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (same_as<double, representation_t<E>>) {
            return native_vector_t<E>{
                .m128_f64 = {__DPL bit_cast<double>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (bfloat16_like<representation_t<E>> ||
            float16_like<representation_t<E>>) {
            return native_vector_t<E>{
                .m128_i16 = {__DPL bit_cast<int16>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (same_as<representation_t<E>, int32>) {
            return native_vector_t<E>{
                .m128i_i32 = {__DPL bit_cast<int32>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, int16>) {
            return native_vector_t<E>{
                .m128i_i16 = {__DPL bit_cast<int16>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, int8>) {
            return native_vector_t<E>{
                .m128i_i8 = {__DPL bit_cast<int8>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, int64>) {
            return native_vector_t<E>{
                .m128i_i64 = {__DPL bit_cast<int64>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, uint32>) {
            return native_vector_t<E>{
                .m128i_u32 = {__DPL bit_cast<uint32>(
                    static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, uint16>) {
            return native_vector_t<E>{
                .m128i_u16 = {__DPL bit_cast<uint16>(
                    static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, uint8>) {
            return native_vector_t<E>{
                .m128i_u8 = {__DPL bit_cast<uint8>(static_cast<E>(args))...},
            };
        } else {
            return native_vector_t<E>{
                .m128i_u64 = {__DPL bit_cast<uint64>(
                    static_cast<E>(args))...},
            };
        }
#endif
    } else {

        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_setr_ps( __DPL bit_cast<float>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_setr_pd( __DPL bit_cast<double>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else if constexpr (sizeof(E) == sizeof(int64)) {
#if __cpp_pack_indexing >= 202311L
            static_assert(sizeof...(Args) == 2);
            return _mm_set_epi64x(
                __DPL bit_cast<int64>(static_cast<E>(args...[1])),
                __DPL bit_cast<int64>(static_cast<E>(args...[0])));
#else
            using array = E[abi_tag::size / sizeof(E)];
            alignas(abi_tag::alignment)
                array buffer{static_cast<E>(__DPL forward<Args>(args))...};
            return __DPL bit_cast<native_vector_t<E>>(buffer);
#endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            return _mm_setr_epi32( __DPL bit_cast<int32>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            return _mm_setr_epi16( __DPL bit_cast<int16>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            return _mm_setr_epi8( __DPL bit_cast<int8>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        }
    }
}

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
constexpr mask<E> initialize(
    abi_tag tag, bitset<simd_abi_traits<E, abi_tag>::size> mask) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    constexpr auto width = simd_abi_traits<E, abi_tag>::size;
    if consteval {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return +xmm::initialize<E>(
                tag, (mask[Is] ? dx::all_bits_v<E> : dx::zero_v<E>)...);
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

DPL_EXPORT template <simd_element E, different_from<abi_tag> Arg,
    core_convertible_to<E>... Args>
requires core_convertible_to<Arg, E> &&
    (different_from<Arg, bool> && ... && different_from<Args, bool>) &&
    (sizeof...(Args) + 1 == simd_abi_traits<E, abi_tag>::size)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> initialize(Arg&& arg, Args&&... args) noexcept {
    return xmm::initialize<E>(
        xmm::abi, __DPL forward<Arg>(arg), __DPL forward<Args>(args)...);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> initialize(
    bitset<simd_abi_traits<E, abi_tag>::size> mask) noexcept {
    return +xmm::initialize<E>(xmm::abi, mask);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif

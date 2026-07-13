// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"

#  if !DPL_MODULES
#    include "dpl/core/type_traits/representation.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/std/concepts/integral_constant_like.h"
#    include "dpl/std/type_traits/conditional.h"     // IWYU pragma: keep
#    include "dpl/std/type_traits/underlying_type.h" // IWYU pragma: keep
#    include "dpl/std/utility/to_signed.h"
#    include "dpl/std/utility/to_underlying.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(vector<E> src, size_t idx) noexcept {
#  if !DPL_COMPILER_MSVC
    struct alignas(abi_tag::alignment) buffer {
        E data[abi_tag::size / sizeof(E)];
    };
    return __DPL bit_cast<E>(__DPL bit_cast<buffer>(src).data[idx]);
#  else
    if constexpr (is_same_v<float, representation_t<E>>) {
        return __DPL bit_cast<E>((+src).m128_f32[idx]);
    } else if constexpr (is_same_v<double, E>) {
        return __DPL bit_cast<E>((+src).m128d_f64[idx]);
    } else if constexpr (is_same_v<ext::float16, E> ||
        is_same_v<ext::bfloat16, E>) {
        return __DPL bit_cast<E>((+src).m128i_i16[idx]);
    } else {
        if constexpr (is_same_v<E, int32>) {
            return __DPL bit_cast<E>((+src).m128i_i32[idx]);
        } else if constexpr (is_same_v<E, int16>) {
            return __DPL bit_cast<E>((+src).m128i_i16[idx]);
        } else if constexpr (is_same_v<E, int8>) {
            return __DPL bit_cast<E>((+src).m128i_i8[idx]);
        } else if constexpr (is_same_v<E, int64>) {
            return __DPL bit_cast<E>((+src).m128i_i64[idx]);
        } else if constexpr (is_same_v<E, uint32>) {
            return __DPL bit_cast<E>((+src).m128i_u32[idx]);
        } else if constexpr (is_same_v<E, uint16>) {
            return __DPL bit_cast<E>((+src).m128i_u16[idx]);
        } else if constexpr (is_same_v<E, uint8>) {
            return __DPL bit_cast<E>((+src).m128i_u8[idx]);
        } else {
            return __DPL bit_cast<E>((+src).m128i_u64[idx]);
        }
    }
#  endif
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(vector<E> src, integral_constant_like auto idx) noexcept {
    if consteval {
        return xmm::extract(src, static_cast<size_t>(idx));
    } else {
        constexpr auto imm8 = static_cast<int>(idx());
        if constexpr (is_same_v<E, float>) {
            return __DPL bit_cast<E>(_mm_extract_ps(+src, imm8));
        } else if constexpr (is_same_v<E, double>) {
            return __DPL bit_cast<E>(
                _mm_extract_epi64(_mm_castpd_si128(+src), idx));
        } else if constexpr (is_same_v<E, ext::float16>) {
#  if DPL_SIMD_X86_AVX512FP16
            if constexpr (same_as<ext::float16, E>) {
                return __DPL bit_cast<E>(static_cast<int16>(
                    _mm_extract_epi16(_mm_castph_si128(+src), imm8)));
            } else {
                return __DPL bit_cast<E>(static_cast<int16>(
                    _mm_extract_epi16(__DPL bit_cast<__m128i>(+src), imm8)));
            }
#  else
            return __DPL bit_cast<E>(static_cast<int16>(
                _mm_extract_epi16(__DPL bit_cast<__m128i>(+src), imm8)));
#  endif
        } else {
            if constexpr (sizeof(E) == sizeof(int64)) {
                return __DPL bit_cast<E>(_mm_extract_epi64(+src, imm8));
            } else if constexpr (sizeof(E) == sizeof(int32)) {
                return __DPL bit_cast<E>(_mm_extract_epi32(+src, imm8));
            } else if constexpr (sizeof(E) == sizeof(int16)) {
                return __DPL bit_cast<E>(
                    static_cast<E>(_mm_extract_epi16(+src, imm8)));
            } else {
                static_assert(sizeof(E) == sizeof(int8));
                return __DPL bit_cast<E>(
                    static_cast<E>(_mm_extract_epi8(+src, imm8)));
            }
        }
    }
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr bool is_true(signed_integral auto val) noexcept {
    return val < 0;
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr bool is_true(unsigned_integral auto val) noexcept {
    return xmm::is_true(__DPL to_signed(val));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
constexpr bool is_true(floating_point auto val) noexcept {
    using int_type = signed_representation_t<decltype(val)>;
    return xmm::is_true(__DPL bit_cast<int_type>(val));
}

template <enumeration T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr bool is_true(T val) noexcept {
    return xmm::is_true(__DPL to_underlying(val));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(mask<E> src, size_t idx) noexcept {
    return xmm::is_true(xmm::extract(vector<E>(+src), idx));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(mask<E> src, integral_constant_like auto idx) noexcept {
    return xmm::is_true(xmm::extract(vector<E>(+src), idx));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(abi_tag, vector<E> src, size_t idx) noexcept
requires requires { xmm::extract(src, idx); }
{
    return xmm::extract(src, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(
    abi_tag, vector<E> src, integral_constant_like auto idx) noexcept
requires requires { xmm::extract(src, idx); }
{
    return xmm::extract(src, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(abi_tag, mask<E> src, size_t idx) noexcept
requires requires { xmm::extract(src, idx); }
{
    return xmm::extract(src, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(
    abi_tag, mask<E> src, integral_constant_like auto idx) noexcept
requires requires { xmm::extract(src, idx); }
{
    return xmm::extract(src, idx);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

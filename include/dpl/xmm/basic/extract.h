// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_element_for.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/conditional.h"     // IWYU pragma: keep
#  include "dpl/std/type_traits/underlying_type.h" // IWYU pragma: keep
#  include "dpl/std/utility/to_signed.h"
#  include "dpl/std/utility/to_underlying.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(abi_tag, simd<E> src, size_t idx) noexcept {
#if !DPL_COMPILER_MSVC
    struct alignas(abi_tag::alignment) buffer {
        E data[abi_tag::size / sizeof(E)];
    };
    return __DPL bit_cast<E>(__DPL bit_cast<buffer>(src).data[idx]);
#else
    if constexpr (same_as<float, lane_representation_t<E>>) {
        return __DPL bit_cast<E>((+src).m128_f32[idx]);
    } else if constexpr (same_as<double, lane_representation_t<E>>) {
        return __DPL bit_cast<E>((+src).m128d_f64[idx]);
    } else if constexpr (bfloat16_like<lane_representation_t<E>> ||
        float16_like<lane_representation_t<E>>) {
        return __DPL bit_cast<E>((+src).m128i_i16[idx]);
    } else {
        if constexpr (same_as<lane_representation_t<E>, int32>) {
            return __DPL bit_cast<E>((+src).m128i_i32[idx]);
        } else if constexpr (same_as<lane_representation_t<E>, int16>) {
            return __DPL bit_cast<E>((+src).m128i_i16[idx]);
        } else if constexpr (same_as<lane_representation_t<E>, int8>) {
            return __DPL bit_cast<E>((+src).m128i_i8[idx]);
        } else if constexpr (same_as<lane_representation_t<E>, int64>) {
            return __DPL bit_cast<E>((+src).m128i_i64[idx]);
        } else if constexpr (same_as<lane_representation_t<E>, uint32>) {
            return __DPL bit_cast<E>((+src).m128i_u32[idx]);
        } else if constexpr (same_as<lane_representation_t<E>, uint16>) {
            return __DPL bit_cast<E>((+src).m128i_u16[idx]);
        } else if constexpr (same_as<lane_representation_t<E>, uint8>) {
            return __DPL bit_cast<E>((+src).m128i_u8[idx]);
        } else {
            return __DPL bit_cast<E>((+src).m128i_u64[idx]);
        }
    }
#endif
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(
    abi_tag tag, simd<E> src, integral_constant_like auto idx) noexcept {
    if consteval {
        return dx::xmm::extract(tag, src, static_cast<size_t>(idx));
    } else {
        constexpr auto imm8 = static_cast<int>(idx);
        if constexpr (same_as<float, lane_representation_t<E>>) {
            return __DPL bit_cast<E>(_mm_extract_ps(+src, imm8));
        } else if constexpr (same_as<double, lane_representation_t<E>>) {
            return __DPL bit_cast<E>(
                _mm_extract_epi64(_mm_castpd_si128(+src), idx));
        } else if constexpr (bfloat16_like<lane_representation_t<E>> ||
            float16_like<lane_representation_t<E>>) {
#if DPL_SIMD_X86_AVX512FP16
            if constexpr (!bfloat16_like<lane_representation_t<E>>) {
                return __DPL bit_cast<E>(static_cast<int16>(
                    _mm_extract_epi16(_mm_castph_si128(+src), imm8)));
            } else {
                return __DPL bit_cast<E>(static_cast<int16>(
                    _mm_extract_epi16(__DPL bit_cast<__m128i>(+src), imm8)));
            }
#else
            return __DPL bit_cast<E>(static_cast<int16>(
                _mm_extract_epi16(__DPL bit_cast<__m128i>(+src), imm8)));
#endif
        } else {
            if constexpr (same_as<lane_representation_t<E>, int32>) {
                return __DPL bit_cast<E>(_mm_extract_epi32(+src, imm8));
            } else if constexpr (same_as<lane_representation_t<E>, int16>) {
                return __DPL bit_cast<E>(
                    static_cast<E>(_mm_extract_epi16(+src, imm8)));
            } else if constexpr (same_as<lane_representation_t<E>, int8>) {
                return __DPL bit_cast<E>(
                    static_cast<E>(_mm_extract_epi8(+src, imm8)));
            } else if constexpr (same_as<lane_representation_t<E>, int64>) {
                return __DPL bit_cast<E>(_mm_extract_epi64(+src, imm8));
            } else if constexpr (same_as<lane_representation_t<E>, uint32>) {
                return __DPL bit_cast<E>(_mm_extract_epi32(+src, imm8));
            } else if constexpr (same_as<lane_representation_t<E>, uint16>) {
                return __DPL bit_cast<E>(
                    static_cast<E>(_mm_extract_epi16(+src, imm8)));
            } else if constexpr (same_as<lane_representation_t<E>, uint8>) {
                return __DPL bit_cast<E>(
                    static_cast<E>(_mm_extract_epi8(+src, imm8)));
            } else {
                return __DPL bit_cast<E>(_mm_extract_epi64(+src, imm8));
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
    return dx::xmm::is_true(__DPL to_signed(val));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
constexpr bool is_true(floating_point auto val) noexcept {
    using int_type = signed_representation_t<decltype(val)>;
    return dx::xmm::is_true(__DPL bit_cast<int_type>(val));
}

template <enumeration T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr bool is_true(T val) noexcept {
    return dx::xmm::is_true(__DPL to_underlying(val));
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(abi_tag tag, mask<E> src, size_t idx) noexcept {
    return dx::xmm::is_true(xmm::extract(tag, simd<E>(+src), idx));
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(
    abi_tag tag, mask<E> src, integral_constant_like auto idx) noexcept {
    return dx::xmm::is_true(xmm::extract(tag, simd<E>(+src), idx));
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(simd<E> src, size_t idx) noexcept {
    return xmm::extract(xmm::abi, src, idx);
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr E extract(simd<E> src, integral_constant_like auto idx) noexcept {
    return xmm::extract(xmm::abi, src, idx);
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(mask<E> src, size_t idx) noexcept {
    return xmm::extract(xmm::abi, src, idx);
}

DPL_EXPORT template <simd_element_for<abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool extract(mask<E> src, integral_constant_like auto idx) noexcept {
    return xmm::extract(xmm::abi, src, idx);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

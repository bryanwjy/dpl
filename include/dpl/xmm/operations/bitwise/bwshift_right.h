// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/bitwise/bwshift_left.h"
#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/utility/template_barrier.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

namespace details {

alignas(16) inline constexpr char niota_epi8[]{112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127};

} // namespace details

template <template_barrier_t = template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E>
    DPL_VECTORCALL bwshift_right(mask<E> val, int shift) noexcept {
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_right(xmm::reinterpret<bit>(val), shift));
    } else {
        auto const idx = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::niota_epi8));
        constexpr auto invalid = static_cast<char>(-1);
        auto const vshift = _mm_set1_epi8(
            shift < mask<E>::size() ? static_cast<char>(shift) : invalid);
        // if msb is set: dst = 0; else: dst = src[idx % 16]
        return _mm_shuffle_epi8(+val, _mm_add_epi8(idx, vshift));
    }
}

template <size_t V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E>
    DPL_VECTORCALL bwshift_right(
        mask<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_right(xmm::reinterpret<bit>(val), imm));
    } else {
        return _mm_bsrli_si128(+val, static_cast<int>(V * sizeof(E)));
    }
}

template <template_barrier_t = template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwshift_right(vector<E> val, int shift) noexcept {
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_right(xmm::reinterpret<bit>(val), shift));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srl_epi64(+val, _mm_cvtsi32_si128(shift));
        } else {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_srav_epi64(
                +val, _mm_set1_epi64x(static_cast<int64>(shift)));
#else
            auto const vval = +val;
            auto const shifted = _mm_srl_epi64(vval, _mm_cvtsi32_si128(shift));
            auto const sign = _mm_cmpgt_epi64(_mm_setzero_si128(), vval);
            auto const fill =
                _mm_sll_epi64(sign, _mm_cvtsi32_si128(64 - shift));
            return _mm_or_si128(shifted, fill);
#endif
        }
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srl_epi32(+val, _mm_cvtsi32_si128(shift));
        } else {
            return _mm_sra_epi32(+val, _mm_cvtsi32_si128(shift));
        }
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srl_epi16(+val, _mm_cvtsi32_si128(shift));
        } else {
            return _mm_sra_epi16(+val, _mm_cvtsi32_si128(shift));
        }
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        auto const vshift = _mm_cvtsi32_si128(shift);
        if constexpr (unsigned_integral<E>) {
            auto const zero = _mm_setzero_si128();
            auto lower = _mm_unpacklo_epi8(vval, zero);
            auto upper = _mm_unpackhi_epi8(vval, zero);
            lower = _mm_srl_epi16(lower, vshift);
            upper = _mm_srl_epi16(upper, vshift);

            return _mm_packus_epi16(lower, upper);
        } else {
            auto const sign = _mm_cmplt_epi8(vval, _mm_setzero_si128());
            auto lower = _mm_unpacklo_epi8(vval, sign);
            auto upper = _mm_unpackhi_epi8(vval, sign);

            lower = _mm_sra_epi16(lower, vshift);
            upper = _mm_sra_epi16(upper, vshift);
            return _mm_packs_epi16(lower, upper);
        }
    }
}

template <size_t V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwshift_right(
        vector<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    static_assert(V >= 0 && V < sizeof(E) * char_bit_v);
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_right(xmm::reinterpret<bit>(val), imm));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srli_epi64(+val, static_cast<int>(V));
        } else {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_srai_epi64(+val, static_cast<int>(V));
#else
            auto const vval = +val;
            auto const shifted = _mm_srli_epi64(vval, static_cast<int>(V));
            auto const sign = _mm_cmpgt_epi64(_mm_setzero_si128(), vval);
            auto const fill = _mm_slli_epi64(sign, static_cast<int>(64 - V));
            return _mm_or_si128(shifted, fill);
#endif
        }
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srli_epi32(+val, static_cast<int>(V));
        } else {
            return _mm_srai_epi32(+val, static_cast<int>(V));
        }
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srli_epi16(+val, static_cast<int>(V));
        } else {
            return _mm_srai_epi16(+val, static_cast<int>(V));
        }
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        if constexpr (unsigned_integral<E>) {
            auto const zero = _mm_setzero_si128();
            auto lower = _mm_unpacklo_epi8(vval, zero);
            auto upper = _mm_unpackhi_epi8(vval, zero);
            lower = _mm_srli_epi16(lower, static_cast<int>(V));
            upper = _mm_srli_epi16(upper, static_cast<int>(V));

            return _mm_packus_epi16(lower, upper);
        } else {
            auto const sign = _mm_cmplt_epi8(vval, _mm_setzero_si128());
            auto lower = _mm_unpacklo_epi8(vval, sign);
            auto upper = _mm_unpackhi_epi8(vval, sign);

            lower = _mm_srai_epi16(lower, static_cast<int>(V));
            upper = _mm_srai_epi16(upper, static_cast<int>(V));
            return _mm_packs_epi16(lower, upper);
        }
    }
}

#if DPL_SIMD_X86_AVX2

template <template_barrier_t = template_barrier, simd_element L,
    common_size_with<L> R>
requires integral<R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<L>
    DPL_VECTORCALL bwshift_right(vector<L> lhs, vector<R> rhs) noexcept {
    if constexpr (!integral<L>) {
        using bit = unsigned_representation_t<L>;
        return xmm::reinterpret<L>(
            xmm::bwshift_right(xmm::reinterpret<bit>(lhs), rhs));
    } else if constexpr (sizeof(L) == sizeof(int64)) {
        if constexpr (unsigned_integral<L>) {
            return _mm_srlv_epi64(+lhs, +rhs);
        } else {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_srav_epi64(+lhs, +rhs);
#  else
            auto const shifted = _mm_srlv_epi64(+lhs, +rhs);
            auto const sign = _mm_cmpgt_epi64(_mm_setzero_si128(), +lhs);
            auto const fill =
                _mm_slli_epi64(sign, _mm_sub_epi64(_mm_set1_epi64x(64), +rhs));
            return _mm_or_si128(shifted, fill);
#  endif
        }
    } else if constexpr (sizeof(L) == sizeof(int32)) {
        if constexpr (unsigned_integral<L>) {
            return _mm_srlv_epi32(+lhs, +rhs);
        } else {
            return _mm_srav_epi32(+lhs, +rhs);
        }
    } else if constexpr (sizeof(L) == sizeof(int16)) {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        if constexpr (unsigned_integral<L>) {
            return _mm_srlv_epi16(+lhs, +rhs);
        } else {
            return _mm_srav_epi16(+lhs, +rhs);
        }
#  else
        if constexpr (unsigned_integral<L>) {
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepu16_epi32(xmm1);
            auto xmm2 = _mm_cvtepu16_epi32(xmm0);
            xmm2 = _mm_srlv_epi32(xmm2, xmm3);
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepu16_epi32(xmm1);
            xmm0 = _mm_cvtepu16_epi32(xmm0);
            xmm0 = _mm_srlv_epi32(xmm0, xmm1);
            xmm1 = _mm_xor_si128(xmm1, xmm1);
            xmm2 = _mm_blend_epi16(xmm1, xmm2, 0x55);
            xmm1 = _mm_blend_epi16(xmm1, xmm0, 0x55);
            return _mm_packus_epi32(xmm2, xmm1);
        } else {
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepi16_epi32(xmm1);
            auto xmm2 = _mm_cvtepi16_epi32(xmm0);
            xmm2 = _mm_srav_epi32(xmm2, xmm3);
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepi16_epi32(xmm1);
            xmm0 = _mm_cvtepi16_epi32(xmm0);
            xmm0 = _mm_srav_epi32(xmm0, xmm1);
            xmm1 = _mm_xor_si128(xmm1, xmm1);
            xmm2 = _mm_blend_epi16(xmm1, xmm2, 0x55);
            xmm1 = _mm_blend_epi16(xmm1, xmm0, 0x55);
            return _mm_packus_epi32(xmm2, xmm1);
        }
#  endif
    } else {
        static_assert(sizeof(L) == sizeof(int8));
        if constexpr (unsigned_integral<L>) {
            using i16 = make_unsigned_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepu8_epi16(xmm1);
            auto xmm2 = _mm_cvtepu8_epi16(xmm0);
            xmm2 = +xmm::bwshift_right(vector<i16>(xmm2), vector<i16>(xmm3));
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepu8_epi16(xmm1);
            xmm0 = _mm_cvtepu8_epi16(xmm0);
            xmm0 = +xmm::bwshift_right(vector<i16>(xmm0), vector<i16>(xmm1));
            xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);
            xmm2 = _mm_and_si128(xmm2, xmm1);
            xmm1 = _mm_and_si128(xmm0, xmm1);
            return _mm_packus_epi16(xmm2, xmm1);
        } else {
            using i16 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepi8_epi16(xmm1);
            auto xmm2 = _mm_cvtepi8_epi16(xmm0);
            xmm2 = +xmm::bwshift_right(simd16(xmm2), simd16(xmm3));
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepi8_epi16(xmm1);
            xmm0 = _mm_cvtepi8_epi16(xmm0);
            xmm0 = +xmm::bwshift_right(simd16(xmm0), simd16(xmm1));
            xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);
            xmm2 = _mm_and_si128(xmm2, xmm1);
            xmm1 = _mm_and_si128(xmm0, xmm1);
            return _mm_packus_epi16(xmm2, xmm1);
        }
    }
}

// No efficient way to right shift without avx2
#endif

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> bwshift_right(abi_tag, mask<E> val, int shift) noexcept
requires requires { xmm::bwshift_right(val, shift); }
{
    return xmm::bwshift_right(val, shift);
}

template <simd_element E, integral_constant_like I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> bwshift_right(abi_tag, mask<E> val, I) noexcept
requires requires { xmm::bwshift_right<I::value>(val); }
{
    return xmm::bwshift_right<I::value>(val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bwshift_right(abi_tag tag, vector<E> val, int shift) noexcept
requires requires { xmm::bwshift_right(val, shift); }
{
    return xmm::bwshift_right(val, shift);
}

template <simd_element E, integral_constant_like I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bwshift_right(abi_tag, vector<E> val, I) noexcept
requires requires { xmm::bwshift_right<I::value>(val); }
{
    return xmm::bwshift_right<I::value>(val);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<L> bwshift_right(abi_tag, vector<L> lhs, vector<R> rhs) noexcept
requires requires { xmm::bwshift_right(lhs, rhs); }
{
    return xmm::bwshift_right(lhs, rhs);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

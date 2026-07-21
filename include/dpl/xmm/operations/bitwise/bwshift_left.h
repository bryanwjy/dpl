// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/core/concepts/common_size_with.h"
#    include "dpl/core/immediate/immediate.h"
#    include "dpl/core/type_traits/representation.h"
#    include "dpl/std/concepts/integral_constant_like.h"
#    include "dpl/std/utility/template_barrier.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

namespace details {
alignas(16) inline constexpr char iota_epi8[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

} // namespace details

template <template_barrier_t = template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E>
    DPL_VECTORCALL bwshift_left(mask<E> val, int shift) noexcept {
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_left(xmm::reinterpret<bit>(val), shift));
    } else {
        auto const idx = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::iota_epi8));
        auto const vshift =
            _mm_set1_epi8(shift < mask<E>::size() ? shift * sizeof(E)
                                                  : sizeof(details::iota_epi8));
        return _mm_shuffle_epi8(+val, _mm_sub_epi8(idx, vshift));
    }
}

template <size_t V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E>
    DPL_VECTORCALL bwshift_left(
        mask<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_left(xmm::reinterpret<bit>(val), imm));
    } else {
        return _mm_bslli_si128(+val, static_cast<int>(V * sizeof(E)));
    }
}

template <template_barrier_t = __DPL template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwshift_left(vector<E> val, int shift) noexcept {
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_left(xmm::reinterpret<bit>(val), shift));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        return _mm_sll_epi64(+val, _mm_cvtsi32_si128(shift));
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_sll_epi32(+val, _mm_cvtsi32_si128(shift));
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_sll_epi16(+val, _mm_cvtsi32_si128(shift));
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto lower = _mm_unpacklo_epi8(vval, zero);
        auto upper = _mm_unpackhi_epi8(vval, zero);

        auto const vshift = _mm_cvtsi32_si128(shift);
        lower = _mm_sll_epi16(lower, vshift);
        upper = _mm_sll_epi16(upper, vshift);

        auto const mask = _mm_set1_epi16(0x00FF);
        lower = _mm_and_si128(lower, mask);
        upper = _mm_and_si128(upper, mask);
        return _mm_packus_epi16(lower, upper);
    }
}

template <size_t V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwshift_left(
        vector<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    static_assert(V >= 0 && V < sizeof(E) * char_bit_v);
    if constexpr (!integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::bwshift_left(xmm::reinterpret<bit>(val), imm));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        return _mm_slli_epi64(+val, static_cast<int>(V));
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_slli_epi32(+val, static_cast<int>(V));
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_slli_epi16(+val, static_cast<int>(V));
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto lower = _mm_unpacklo_epi8(vval, zero);
        auto upper = _mm_unpackhi_epi8(vval, zero);

        constexpr int shift = static_cast<int>(V);
        lower = _mm_slli_epi16(lower, shift);
        upper = _mm_slli_epi16(upper, shift);

        auto const mask = _mm_set1_epi16(0x00FF);
        lower = _mm_and_si128(lower, mask);
        upper = _mm_and_si128(upper, mask);
        return _mm_packus_epi16(lower, upper);
    }
}

#  if DPL_SIMD_X86_AVX2

template <template_barrier_t = __DPL template_barrier, simd_element L,
    common_size_with<L> R>
requires integral<R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<L>
    DPL_VECTORCALL bwshift_left(vector<L> lhs, vector<R> rhs) noexcept {
    if constexpr (!integral<L>) {
        using bit = unsigned_representation_t<L>;
        return xmm::reinterpret<L>(
            xmm::bwshift_left(xmm::reinterpret<bit>(lhs), rhs));
    } else if constexpr (sizeof(L) == sizeof(int64)) {
        return _mm_sllv_epi64(+lhs, +rhs);
    } else if constexpr (sizeof(L) == sizeof(int32)) {
        return _mm_sllv_epi32(+lhs, +rhs);
    } else if constexpr (sizeof(L) == sizeof(int16)) {
#    if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        return _mm_sllv_epi16(+lhs, +rhs);
#    else
        // Maybe just use the ymm registers?
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;

        auto xmm3 = _mm_cvtepi16_epi32(xmm1);
        auto xmm2 = _mm_cvtepi16_epi32(xmm0);
        xmm2 = _mm_sllv_epi32(xmm2, xmm3);

        xmm1 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm0, 8));
        xmm0 = _mm_sllv_epi32(xmm0, xmm1);
        xmm1 = _mm_xor_si128(xmm1, xmm1); // zero
        constexpr auto mask = 0b01010101;
        xmm2 = _mm_blend_epi16(xmm1, xmm2, mask);
        xmm1 = _mm_blend_epi16(xmm1, xmm0, mask);
        return _mm_packus_epi32(xmm2, xmm1);
#    endif
    } else {
        static_assert(sizeof(L) == sizeof(int8));
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;
        using i16 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
        auto xmm3 = _mm_cvtepi8_epi16(xmm1);
        auto xmm2 = _mm_cvtepi8_epi16(xmm0);
        xmm2 = +xmm::bwshift_left(vector<i16>(xmm2), vector<i16>(xmm3));

        xmm1 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm0, 8));
        xmm0 = +xmm::bwshift_left(vector<i16>(xmm0), vector<i16>(xmm1));
        // broadcast 0x00ff00ff
        xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);

        xmm2 = _mm_and_si128(xmm2, xmm1);
        xmm1 = _mm_and_si128(xmm0, xmm1);
        return _mm_packus_epi16(xmm2, xmm1);
    }
}

#  else
// SSE 4.2
template <template_barrier_t = __DPL template_barrier, simd_element L,
    common_size_with<L> R>
requires integral<R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<L>
    DPL_VECTORCALL bwshift_left(vector<L> lhs, vector<R> rhs) noexcept {
    if constexpr (!integral<L>) {
        using bit = unsigned_representation_t<L>;
        return xmm::reinterpret<L>(
            xmm::bwshift_left(xmm::reinterpret<bit>(lhs), rhs));
    } else if constexpr (sizeof(L) == sizeof(int64)) {
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;
        auto xmm2 = _mm_sll_epi64(xmm0, xmm1);
        xmm1 = _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(3, 2, 3, 2));
        xmm0 = _mm_sll_epi64(xmm0, xmm1);
        return _mm_castpd_si128(
            _mm_move_sd(_mm_castsi128_pd(xmm0), _mm_castsi128_pd(xmm2)));
    } else if constexpr (sizeof(L) == sizeof(int32)) {
        constexpr auto magic = 1065353216;
        auto xmm0 = +lhs;
        auto xmm1 = _mm_slli_epi32(+rhs, 23);
        xmm1 = _mm_add_epi32(xmm1, _mm_set1_epi32(magic));
        xmm1 = _mm_cvttps_epi32(_mm_castsi128_ps(xmm1));
        xmm1 = _mm_and_si128(_mm_cmplt_epi32(rhs, _mm_set1_epi32(32)), xmm1);
        return _mm_mullo_epi32(xmm0, xmm1);
    } else if constexpr (sizeof(L) == sizeof(int16)) {
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;

        auto xmm3 = _mm_cvtepi16_epi32(xmm1);
        auto xmm2 = _mm_cvtepi16_epi32(xmm0);
        using i32 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
        xmm2 = +xmm::bwshift_left(vector<i32>(xmm2), vector<i32>(xmm3));

        xmm1 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm0, 8));
        xmm0 = +xmm::bwshift_left(vector<i32>(xmm0), vector<i32>(xmm1));

        xmm1 = _mm_xor_si128(xmm1, xmm1); // zero
        constexpr auto mask = 0b01010101;
        xmm2 = _mm_blend_epi16(xmm1, xmm2, mask);
        xmm1 = _mm_blend_epi16(xmm1, xmm0, mask);
        return _mm_packus_epi32(xmm2, xmm1);
    } else {
        static_assert(sizeof(L) == sizeof(int8));
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;

        using i16 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;

        auto xmm3 = _mm_cvtepi8_epi16(xmm1);
        auto xmm2 = _mm_cvtepi8_epi16(xmm0);
        xmm2 = +bwshift_left(vector<i16>(xmm2), vector<i16>(xmm3));

        xmm1 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm0, 8));
        xmm0 = +bwshift_left(vector<i16>(xmm0), vector<i16>(xmm1));

        // broadcast 0x00ff00ff
        xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);

        xmm2 = _mm_and_si128(xmm2, xmm1);
        xmm1 = _mm_and_si128(xmm0, xmm1);
        return _mm_packus_epi16(xmm2, xmm1);
    }
}

#  endif

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> bwshift_left(abi_tag, mask<E> val, int shift) noexcept
requires requires { xmm::bwshift_left(val, shift); }
{
    return xmm::bwshift_left(val, shift);
}

template <simd_element E, integral_constant_like I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> bwshift_left(abi_tag, mask<E> val, I) noexcept
requires requires { xmm::bwshift_left<I::value>(val); }
{
    return xmm::bwshift_left<I::value>(val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bwshift_left(abi_tag, vector<E> val, int shift) noexcept
requires requires { xmm::bwshift_left(val, shift); }
{
    return xmm::bwshift_left(val, shift);
}

template <simd_element E, integral_constant_like I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bwshift_left(abi_tag, vector<E> val, I) noexcept
requires requires { xmm::bwshift_left<I::value>(val); }
{
    return xmm::bwshift_left<I::value>(val);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<L> bwshift_left(abi_tag, vector<L> lhs, vector<R> rhs) noexcept
requires requires { xmm::bwshift_left(lhs, rhs); }
{
    return xmm::bwshift_left(lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

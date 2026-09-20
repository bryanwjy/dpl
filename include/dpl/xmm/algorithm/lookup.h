// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/operations/bitwise.h"
#    include "dpl/xmm/operations/permute.h"
#    include "dpl/xmm/operations/reinterpret.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

namespace details {
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8> DPL_VECTORCALL to_byte_shuffle_idx(
    vector<int8> idx) noexcept {
    return _mm_adds_epu8(+idx, +xmm::broadcast<int8>(0x70));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16> DPL_VECTORCALL to_byte_shuffle_idx(
    vector<int16> idx) noexcept {
    idx = xmm::bwor(idx, xmm::broadcast<int16>(0x80));
    return xmm::bwor(
        xmm::bwshift_left(idx, imm<9zu>), xmm::bwshift_left(idx, imm<1zu>));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32> DPL_VECTORCALL to_byte_shuffle_idx(
    vector<int32> idx) noexcept {
    idx = xmm::bwshift_left(idx, imm<2zu>);
    idx = xmm::reinterpret<int32>(xmm::permute(
        xmm::reinterpret<int8>(idx), details::widen_iota<4, int32>()));
    return xmm::add(idx, xmm::broadcast<int32>(0x03020100));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int16> DPL_VECTORCALL is_byte_idx_oob(vector<int16> idx) noexcept {
    return xmm::cmpeq(
        vector<int16>(_mm_subs_epu16(+xmm::broadcast<int16>(8), +idx)),
        xmm::broadcast<int16>(dx::zero));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int32> DPL_VECTORCALL is_byte_idx_oob(vector<int32> idx) noexcept {
    constexpr auto bias4 = static_cast<int32>(0x80000003);
    auto const bias = xmm::broadcast<int32>(dx::msb);
    return xmm::cmplt(xmm::broadcast<int32>(bias4), xmm::bwxor(bias, idx));
}
} // namespace details

template <sized_element<1zu> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, vector<int8> rhs, dx::zero_t) noexcept {
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(
            xmm::lookup(xmm::reinterpret<int8>(lhs), rhs, zero));
    } else {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        auto const inrange = _mm_cmp_epu8_mask(
            +rhs, +xmm::broadcast<int8>(vector<E>::size()), _MM_CMPINT_LT);
        return _mm_maskz_shuffle_epi8(inrange, +lhs, +rhs);
#  else
        return _mm_shuffle_epi8(+lhs, +details::to_byte_shuffle_idx(rhs));
#  endif
    }
}

template <sized_element<2zu> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, vector<int16> rhs, dx::zero_t zero) noexcept {
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(
            xmm::lookup(xmm::reinterpret<int16>(lhs), rhs, zero));
    } else {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        auto const oob = details::is_byte_idx_oob(rhs);
        auto const inrange = _mm_cmp_epu8_mask(
            +oob, +xmm::broadcast<int8>(dx::zero), _MM_CMPINT_EQ);
        auto const idx = details::to_byte_shuffle_idx(rhs);
        return _mm_maskz_shuffle_epi8(inrange, +lhs, +idx);
#  else
        auto const idx = details::to_byte_shuffle_idx(rhs);
        auto const oob = details::is_byte_idx_oob(rhs);
        return _mm_shuffle_epi8(+lhs, +xmm::select(oob, dx::all_bits, idx));
#  endif
    }
}

template <sized_element<4zu> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, vector<int32> rhs, dx::zero_t zero) noexcept {
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(
            xmm::lookup(xmm::reinterpret<int32>(lhs), rhs, zero));
    } else {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        auto const oob = details::is_byte_idx_oob(rhs);
        auto const inrange = _mm_cmp_epu8_mask(
            +oob, +xmm::broadcast<int8>(dx::zero), _MM_CMPINT_EQ);
        auto const idx = details::to_byte_shuffle_idx(rhs);
        return _mm_maskz_shuffle_epi8(inrange, +lhs, +idx);
#  else
        auto const idx = details::to_byte_shuffle_idx(rhs);
        auto const oob = details::is_byte_idx_oob(rhs);
        return _mm_shuffle_epi8(+lhs, +xmm::select(oob, dx::all_bits, idx));
#  endif
    }
}

template <sized_element<1zu> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, ssize_vector_t<E> rhs, vector<E> src) noexcept {
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(xmm::lookup(
            xmm::reinterpret<int8>(lhs), rhs, xmm::reinterpret<int8>(src)));
    } else {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        auto const inrange = _mm_cmp_epu8_mask(
            +rhs, +xmm::broadcast<int8>(vector<E>::size()), _MM_CMPINT_LT);
        return _mm_mask_shuffle_epi8(+src, inrange, +lhs, +rhs);
#  else
        auto const idx = details::to_byte_shuffle_idx(rhs);
        auto const oob = dx::bwshift_right(idx, 7zu);
        return xmm::bwor(vector<E>(_mm_shuffle_epi8(+lhs, +idx)),
            xmm::bwand(xmm::reinterpret<E>(oob), src));
#  endif
    }
}

template <sized_element<2zu> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, ssize_vector_t<E> rhs, vector<E> src) noexcept {
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(xmm::lookup(
            xmm::reinterpret<int16>(lhs), rhs, xmm::reinterpret<int16>(src)));
    } else {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        auto const oob = details::is_byte_idx_oob(rhs);
        auto const inrange = _mm_cmp_epu8_mask(
            +rhs, +xmm::broadcast<int8>(dx::zero), _MM_CMPINT_EQ);
        auto const idx = details::to_byte_shuffle_idx(rhs);
        return _mm_mask_shuffle_epi8(+src, inrange, +lhs, +idx);
#  else
        auto const oob = details::is_byte_idx_oob(rhs);
        auto const idx =
            xmm::select(oob, dx::all_bits, details::to_byte_shuffle_idx(rhs));
        src = xmm::select(oob, src, dx::zero);
        return xmm::bwor(vector<E>(_mm_shuffle_epi8(+lhs, +idx)), src);
#  endif
    }
}

template <sized_element<4zu> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, ssize_vector_t<E> rhs, vector<E> src) noexcept {
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(xmm::lookup(
            xmm::reinterpret<int32>(lhs), rhs, xmm::reinterpret<int32>(src)));
    } else {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        auto const oob = details::is_byte_idx_oob(rhs);
        auto const inrange = _mm_cmp_epu8_mask(
            +rhs, +xmm::broadcast<int8>(dx::zero), _MM_CMPINT_EQ);
        auto const idx = details::to_byte_shuffle_idx(rhs);
        return _mm_mask_shuffle_epi8(+src, inrange, +lhs, +idx);
#  else
        auto const oob = details::is_byte_idx_oob(rhs);
        auto const idx =
            xmm::select(oob, dx::all_bits, details::to_byte_shuffle_idx(rhs));
        src = xmm::select(oob, src, dx::zero);
        return dx::bwor(vector<E>(_mm_shuffle_epi8(+lhs, +idx)), src);
#  endif
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, size_vector_t<E> rhs, dx::zero_t zero) noexcept {
    return xmm::lookup(
        lhs, xmm::reinterpret<signed_representation_t<E>>(rhs), zero);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL lookup(
        vector<E> lhs, size_vector_t<E> rhs, vector<E> src) noexcept {
    return xmm::lookup(
        lhs, xmm::reinterpret<signed_representation_t<E>>(rhs), src);
}

template <simd_element E, integral I>
requires common_size_with<E, I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> lookup(
    abi_tag, vector<E> lhs, vector<I> idx, dx::zero_t zero) noexcept
requires requires { xmm::lookup(lhs, idx, zero); }
{
    return xmm::lookup(lhs, idx, zero);
}

template <simd_element E, integral I>
requires common_size_with<E, I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> lookup(abi_tag, vector<E> lhs, vector<I> idx,
    type_identity_t<vector<E>> src) noexcept
requires requires { xmm::lookup(lhs, idx, src); }
{
    return xmm::lookup(lhs, idx, src);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif // if DPL_SIMD_X86_SSE4_2

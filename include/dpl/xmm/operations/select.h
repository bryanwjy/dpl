// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/core/concepts/common_size_with.h"
#    include "dpl/core/immediate/const_mask.h"
#    include "dpl/core/type_traits/common_size_type.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/from_bitset.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

template <simd_element C, simd_element E>
requires common_size_with<C, E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL select(
        mask<C> condition, vector<E> lhs, vector<E> rhs) noexcept {
    if constexpr (same_as<E, float>) {
        return _mm_blendv_ps(+rhs, +lhs, +xmm::reinterpret<float>(condition));
    } else if constexpr (same_as<E, double>) {
        return _mm_blendv_pd(+rhs, +lhs, +xmm::reinterpret<double>(condition));
    } else if constexpr (common_size_with<E, float>) {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<float>(lhs), xmm::reinterpret<float>(rhs)));
    } else if constexpr (common_size_with<E, double>) {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<double>(lhs), xmm::reinterpret<double>(rhs)));
    } else if constexpr (common_size_with<E, int16>) {
        return xmm::reinterpret<E>(vector<int16>(_mm_blendv_epi8(
            +xmm::reinterpret<int16>(rhs), +xmm::reinterpret<int16>(lhs),
            +xmm::reinterpret<int16>(condition))));
    } else {
        static_assert(common_size_with<E, int8>);
        return xmm::reinterpret<E>(vector<int8>(_mm_blendv_epi8(
            +xmm::reinterpret<int8>(rhs), +xmm::reinterpret<int8>(lhs),
            +xmm::reinterpret<int8>(condition))));
    }
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<R>
    DPL_VECTORCALL select(mask<L> lhs, dx::zero_t, vector<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(vector<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<R>
    DPL_VECTORCALL select(mask<L> lhs, vector<R> rhs, dx::zero_t) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_and_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(vector<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<R>
    DPL_VECTORCALL select(mask<L> lhs, dx::all_bits_t, vector<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_or_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(vector<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<R>
    DPL_VECTORCALL select(mask<L> lhs, vector<R> rhs, dx::all_bits_t) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const all = [](__m128i val) { return _mm_cmpeq_epi32(val, val); }(
                         _mm_undefined_si128());
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(lhs)),
            +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(vector<bit>(result));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL select(
    abi_tag, mask<L> lhs, dx::zero_t zero, vector<R> rhs) noexcept
requires requires { xmm::select(lhs, zero, rhs); }
{
    return xmm::select(lhs, zero, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL select(
    abi_tag, mask<L> lhs, vector<R> rhs, dx::zero_t zero) noexcept
requires requires { xmm::select(lhs, rhs, zero); }
{
    return xmm::select(lhs, rhs, zero);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL select(
    abi_tag, mask<L> lhs, dx::all_bits_t all_bits, vector<R> rhs) noexcept
requires requires { xmm::select(lhs, all_bits, rhs); }
{
    return xmm::select(lhs, all_bits, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL select(
    abi_tag, mask<L> lhs, vector<R> rhs, dx::all_bits_t all_bits) noexcept
requires requires { xmm::select(lhs, rhs, all_bits); }
{
    return xmm::select(lhs, rhs, all_bits);
}

template <simd_element C, simd_element L, simd_element R>
requires common_size_with<L, R> && common_size_with<L, C> &&
    common_size_with<R, C>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>>
    DPL_VECTORCALL select(
        mask<C> condition, mask<L> lhs, mask<R> rhs) noexcept {
    using E = common_size_type_t<L, R>;
    return +xmm::select(condition, vector<E>(+xmm::reinterpret<E>(lhs)),
        vector<E>(+xmm::reinterpret<E>(rhs)));
}

template <imask2_t V, sized_element<8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E> select(
    cmask2_t<V> condition, vector<E> lhs, vector<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<double, E>) {
        return _mm_blend_pd(+rhs, +lhs, imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<double>(lhs), xmm::reinterpret<double>(rhs)));
    }
}

template <imask4_t V, sized_element<4> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E> select(
    cmask4_t<V> condition, vector<E> lhs, vector<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<float, E>) {
        return _mm_blend_ps(+rhs, +lhs, imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<float>(lhs), xmm::reinterpret<float>(rhs)));
    }
}

template <imask8_t V, sized_element<2> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E> select(
    cmask8_t<V> condition, vector<E> lhs, vector<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<int16, E>) {
        return _mm_blend_epi16(+rhs, +lhs, imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<int16>(lhs), xmm::reinterpret<int16>(rhs)));
    }
}

template <imask16_t V, sized_element<1> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E> select(
    cmask16_t<V> condition, vector<E> lhs, vector<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<int8, E>) {
        constexpr auto imm = xmm::from_bitset<E>(bitset<16>(V));
        return _mm_blendv_epi8(+rhs, +lhs, +imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<int8>(lhs), xmm::reinterpret<int8>(rhs)));
    }
}

template <simd_element L, common_size_with<L> R, imask_t<L> V>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>> select(
    cmask_t<L, V> condition, mask<L> lhs, mask<R> rhs) noexcept {
    using E = common_size_type_t<L, R>;
    return +xmm::select(condition, vector<E>(+xmm::reinterpret<E>(lhs)),
        vector<E>(+xmm::reinterpret<E>(rhs)));
}

template <integral auto V, simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(mask<L> lhs, mask<R> rhs) noexcept {
    return xmm::select(cmask_t<L, V>(), lhs, rhs);
}

template <integral auto V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(vector<E> lhs, vector<E> rhs) noexcept {
    return xmm::select(cmask_t<E, V>(), lhs, rhs);
}

template <simd_element C, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(
    abi_tag tag, mask<C> condition, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::select(condition, lhs, rhs); }
{
    return xmm::select(condition, lhs, rhs);
}

template <simd_element C, simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(
    abi_tag tag, mask<C> condition, mask<L> lhs, mask<R> rhs) noexcept
requires requires { xmm::select(condition, lhs, rhs); }
{
    return xmm::select(condition, lhs, rhs);
}

template <simd_element L, common_size_with<L> R, imask_t<L> V>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(
    abi_tag tag, cmask_t<L, V> condition, mask<L> lhs, mask<R> rhs) noexcept {
    return xmm::select(condition, lhs, rhs);
}

template <simd_element E, imask_t<E> V>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E> select(abi_tag tag, cmask_t<E, V> condition, vector<E> lhs,
    vector<E> rhs) noexcept {
    return xmm::select(condition, lhs, rhs);
}

template <simd_element C, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bit_drop(mask<C> mask, vector<E> val) noexcept
requires requires { xmm::select(mask, dx::zero, val); }
{
    return xmm::select(mask, dx::zero, val);
}

template <simd_element C, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bit_keep(mask<C> mask, vector<E> val) noexcept
requires requires { xmm::select(mask, val, dx::zero); }
{
    return xmm::select(mask, val, dx::zero);
}

template <simd_element C, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bit_fill(mask<C> mask, vector<E> val) noexcept
requires requires { xmm::select(mask, dx::all_bits, val); }
{
    return xmm::select(mask, dx::all_bits, val);
}

template <simd_element C, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bit_spill(mask<C> mask, vector<E> val) noexcept
requires requires { xmm::select(mask, val, dx::all_bits); }
{
    return xmm::select(mask, val, dx::all_bits);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

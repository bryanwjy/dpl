// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

DPL_EXPORT template <simd_element C, simd_element E>
requires common_size_with<C, E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL select(
        mask<C> condition, simd<E> lhs, simd<E> rhs) noexcept {
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
        return xmm::reinterpret<E>(simd<int16>(_mm_blendv_epi8(
            +xmm::reinterpret<int16>(rhs), +xmm::reinterpret<int16>(lhs),
            +xmm::reinterpret<int16>(condition))));
    } else {
        static_assert(common_size_with<E, int8>);
        return xmm::reinterpret<E>(simd<int8>(_mm_blendv_epi8(
            +xmm::reinterpret<int8>(rhs), +xmm::reinterpret<int8>(lhs),
            +xmm::reinterpret<int8>(condition))));
    }
}

DPL_EXPORT template <simd_element C, simd_element L, simd_element R>
requires common_size_with<L, R> && common_size_with<L, C> &&
    common_size_with<R, C>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>>
    DPL_VECTORCALL select(
        mask<C> condition, mask<L> lhs, mask<R> rhs) noexcept {
    using E = common_size_type_t<L, R>;
    return +xmm::select(condition, simd<E>(+xmm::reinterpret<E>(lhs)),
        simd<E>(+xmm::reinterpret<E>(rhs)));
}

DPL_EXPORT template <bit_type_t<2> V, sized_element<8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> select(
    const_mask<2, V> condition, simd<E> lhs, simd<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<double, E>) {
        return _mm_blend_pd(+rhs, +lhs, imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<double>(lhs), xmm::reinterpret<double>(rhs)));
    }
}

DPL_EXPORT template <bit_type_t<4> V, sized_element<4> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> select(
    const_mask<4, V> condition, simd<E> lhs, simd<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<float, E>) {
        return _mm_blend_ps(+rhs, +lhs, imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<float>(lhs), xmm::reinterpret<float>(rhs)));
    }
}

DPL_EXPORT template <bit_type_t<8> V, sized_element<2> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> select(
    const_mask<8, V> condition, simd<E> lhs, simd<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<int16, E>) {
        return _mm_blend_epi16(+rhs, +lhs, imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<int16>(lhs), xmm::reinterpret<int16>(rhs)));
    }
}

DPL_EXPORT template <bit_type_t<16> V, sized_element<1> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> select(
    const_mask<16, V> condition, simd<E> lhs, simd<E> rhs) noexcept {
    constexpr auto imm = static_cast<int>(condition());
    if constexpr (same_as<int8, E>) {
        constexpr auto imm = xmm::initialize<E>(bitset<16>(V));
        return _mm_blendv_epi8(+rhs, +lhs, +imm);
    } else {
        return xmm::reinterpret<E>(xmm::select(condition,
            xmm::reinterpret<int8>(lhs), xmm::reinterpret<int8>(rhs)));
    }
}

DPL_EXPORT template <simd_element L, simd_element R,
    bit_type_t<simd_abi_traits<abi_tag, L>::size> V>
requires common_size_with<L, R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>> select(
    const_mask<simd_abi_traits<abi_tag, L>::size, V> condition, mask<L> lhs,
    mask<R> rhs) noexcept {
    using E = common_size_type_t<L, R>;
    return +xmm::select(condition, simd<E>(+xmm::reinterpret<E>(lhs)),
        simd<E>(+xmm::reinterpret<E>(rhs)));
}

DPL_EXPORT template <integral auto V, simd_element L, simd_element R>
requires common_size_with<L, R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(mask<L> lhs, mask<R> rhs) noexcept {
    using mask_type = const_mask<simd_abi_traits<abi_tag, L>::size, V>;
    return xmm::select(mask_type(), lhs, rhs);
}

DPL_EXPORT template <integral auto V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(simd<E> lhs, simd<E> rhs) noexcept {
    using mask_type = const_mask<simd_abi_traits<abi_tag, E>::size, V>;
    return xmm::select(mask_type(), lhs, rhs);
}

DPL_EXPORT template <simd_element C, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(
    abi_tag tag, mask<C> condition, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::select(condition, lhs, rhs); }
{
    return xmm::select(condition, lhs, rhs);
}

DPL_EXPORT template <simd_element C, simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline auto select(
    abi_tag tag, mask<C> condition, mask<L> lhs, mask<R> rhs) noexcept
requires requires { xmm::select(condition, lhs, rhs); }
{
    return xmm::select(condition, lhs, rhs);
}

DPL_EXPORT template <simd_element L, simd_element R,
    bit_type_t<simd_abi_traits<abi_tag, L>::size> V>
requires common_size_with<L, R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>> select(abi_tag tag,
    const_mask<simd_abi_traits<abi_tag, L>::size, V> condition, mask<L> lhs,
    mask<R> rhs) noexcept {
    using E = common_size_type_t<L, R>;
    return xmm::select(condition, lhs, rhs);
}

DPL_EXPORT template <simd_element E,
    bit_type_t<simd_abi_traits<abi_tag, E>::size> V>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> select(abi_tag tag,
    const_mask<simd_abi_traits<abi_tag, E>::size, V> condition, simd<E> lhs,
    simd<E> rhs) noexcept {
    return xmm::select(condition, lhs, rhs);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

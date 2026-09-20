// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/operations/arithmetic.h"
#    include "dpl/xmm/operations/bitwise.h"
#    include "dpl/xmm/operations/reinterpret.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

namespace details {
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<int8> left_shift_idx() noexcept {
    return []<size_t... Is>(index_sequence<Is...>) {
        constexpr auto offset = 0x70;
        alignas(16) constexpr int8 array[] = {(Is + offset)...};
        return xmm::load<int8>(dx::aligned, array);
    }(make_index_sequence<16>{});
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<int8> right_shift_idx() noexcept {
    return []<size_t... Is>(index_sequence<Is...>) {
        alignas(16) constexpr int8 array[] = {Is...};
        return xmm::load<int8>(dx::aligned, array);
    }(make_index_sequence<16>{});
}
} // namespace details

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL shift_left(vector<E> lhs, size_t count) noexcept {
    using idx_t = signed_representation_t<E>;
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(
            xmm::shift_left(xmm::reinterpret<idx_t>(lhs), count));
    } else {
        auto const idx = [count]() -> vector<int8> {
            constexpr auto size = vector<E>::size();
            auto const vidx = details::left_shift_idx();
            auto const count8 =
                static_cast<int8>(count < size ? count : size) * sizeof(E);
            auto const vcount = xmm::broadcast<int8>(count8);
            return xmm::add(vidx, vcount);
        }();
        return _mm_shuffle_epi8(+lhs, +idx);
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL shift_right(vector<E> lhs, size_t count) noexcept {
    using idx_t = signed_representation_t<E>;
    if constexpr (!integral<E>) {
        return xmm::reinterpret<E>(
            xmm::shift_right(xmm::reinterpret<idx_t>(lhs), count));
    } else {
        auto const idx = [count]() -> vector<int8> {
            constexpr auto size = vector<E>::size();
            auto const vidx = details::right_shift_idx();
            auto const count8 =
                static_cast<int8>(count < size ? count : size) * sizeof(E);
            auto const vcount = xmm::broadcast<int8>(count8);
            return xmm::subtract(vidx, vcount);
        }();

        return _mm_shuffle_epi8(+lhs, +idx);
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(
        vector<E> lhs, vector<E> rhs, size_t count) noexcept {
    constexpr auto size = vector<E>::size();
    count = count < size ? count : size;
    auto const high = xmm::shift_left(lhs, count);
    auto const low = xmm::shift_right(rhs, size - count);
    return xmm::bwor(low, high);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(vector<E> src, cmask_t<E, M> mask, vector<E> lhs,
        vector<E> rhs, size_t count) noexcept
requires requires { xmm::bwor(src, mask, lhs, rhs); }
{
    constexpr auto size = vector<E>::size();
    count = count < size ? count : size;
    auto const high = xmm::shift_left(lhs, count);
    auto const low = xmm::shift_right(rhs, size - count);
    return xmm::bwor(src, mask, low, high);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(dx::zero_t zero, cmask_t<E, M> mask,
        vector<E> lhs, vector<E> rhs, size_t count) noexcept
requires requires { xmm::bwor(zero, mask, lhs, rhs); }
{
    constexpr auto size = vector<E>::size();
    count = count < size ? count : size;
    auto const high = xmm::shift_left(lhs, count);
    auto const low = xmm::shift_right(rhs, size - count);
    return xmm::bwor(zero, mask, low, high);
}

template <size_t N, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(
        vector<E> lhs, vector<E> rhs, immediate<N> count = imm<N>) noexcept {
    if constexpr (N == 0) {
        return lhs;
    } else if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(
            xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs), count));
    } else
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_alignr_epi32(+rhs, +lhs, static_cast<int>(N));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        return _mm_alignr_epi64(+rhs, +lhs, static_cast<int>(N));
    } else
#  endif // if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    {
        constexpr auto imm8 = static_cast<int>(N * sizeof(E));
#  if DPL_SIMD_X86_SSE3
        return _mm_alignr_epi8(+rhs, +lhs, imm8);
#  else
        constexpr auto size = static_cast<int>(vector<E>::size());
        return _mm_or_si128(
            _mm_srli_si128(+lhs, imm8), _mm_slli_si128(+rhs, size - imm8));
#  endif // if DPL_SIMD_X86_SSE3
    }
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512F
template <sized_element<sizeof(int32)> E, imask_t<E> M,
    integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(vector<E> src, cmask_t<int32, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(xmm::reinterpret<int_t>(src),
            mask, xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs),
            count));
    } else {
        return _mm_mask_alignr_epi32(+src, M, +rhs, +lhs, static_cast<int>(N));
    }
}

template <sized_element<sizeof(int64)> E, imask_t<E> M,
    integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(vector<E> src, cmask_t<int64, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(xmm::reinterpret<int_t>(src),
            mask, xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs),
            count));
    } else {
        return _mm_mask_alignr_epi64(+src, M, +rhs, +lhs, static_cast<int>(N));
    }
}

template <sized_element<sizeof(int32)> E, imask_t<E> M,
    integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(dx::zero_t zero, cmask_t<int32, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(zero, mask,
            xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs), count));
    } else {
        return _mm_maskz_alignr_epi32(M, +rhs, +lhs, static_cast<int>(N));
    }
}

template <sized_element<sizeof(int64)> E, imask_t<E> M,
    integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(dx::zero_t zero, cmask_t<int64, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(zero, mask,
            xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs), count));
    } else {
        return _mm_maskz_alignr_epi64(M, +rhs, +lhs, static_cast<int>(N));
    }
}
#    endif // if DPL_SIMD_X86_AVX512F

#    if DPL_SIMD_X86_AVX512BW
template <sized_element<sizeof(int8)> E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(vector<E> src, cmask_t<int8, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(xmm::reinterpret<int_t>(src),
            mask, xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs),
            count));
    } else {
        return _mm_mask_alignr_epi8(+src, M, +rhs, +lhs, static_cast<int>(N));
    }
}

template <sized_element<sizeof(int8)> E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(dx::zero_t zero, cmask_t<int8, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(zero, mask,
            xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs), count));
    } else {
        return _mm_maskz_alignr_epi8(M, +rhs, +lhs, static_cast<int>(N));
    }
}

namespace details {
consteval uint16 widen_bits16(uint8 m) {
    uint32 x = m;
    x = (x | (x << 4)) & 0x0F0Fu;
    x = (x | (x << 2)) & 0x3333u;
    x = (x | (x << 1)) & 0x5555u;
    return static_cast<uint16>(x * 3);
}
} // namespace details

template <sized_element<sizeof(int16)> E, imask_t<E> M,
    integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(vector<E> src, cmask_t<int16, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(xmm::reinterpret<int_t>(src),
            mask, xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs),
            count));
    } else {
        return _mm_mask_alignr_epi8(+src, details::widen_bits16(M), +rhs, +lhs,
            static_cast<int>(N * 2));
    }
}

template <sized_element<sizeof(int16)> E, imask_t<E> M,
    integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL slide_left(dx::zero_t zero, cmask_t<int16, M> mask,
        vector<E> lhs, vector<E> rhs, N count) noexcept {
    if constexpr (!integral<E>) {
        using int_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(xmm::slide_left(zero, mask,
            xmm::reinterpret<int_t>(lhs), xmm::reinterpret<int_t>(rhs), count));
    } else {
        return _mm_maskz_alignr_epi8(
            details::widen_bits16(M), +rhs, +lhs, static_cast<int>(N * 2));
    }
}
#    endif // if DPL_SIMD_X86_AVX512BW
#  endif   // if DPL_SIMD_X86_AVX512VL

template <simd_element E, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_left(
    abi_tag, vector<E> lhs, vector<E> rhs, N count) noexcept
requires requires { xmm::slide_left<N::value>(lhs, rhs); }
{
    return xmm::slide_left<N::value>(lhs, rhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_left(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs, N count) noexcept
requires requires { xmm::slide_left(src, mask, lhs, rhs, count); }
{
    return xmm::slide_left(src, mask, lhs, rhs, count);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_left(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, vector<E> rhs, N count) noexcept
requires requires { xmm::slide_left(zero, mask, lhs, rhs, count); }
{
    return xmm::slide_left(zero, mask, lhs, rhs, count);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_left(
    abi_tag, vector<E> lhs, vector<E> rhs, size_t count) noexcept
requires requires { xmm::slide_left(lhs, rhs, count); }
{
    return xmm::slide_left(lhs, rhs, count);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_left(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs, size_t count) noexcept
requires requires { xmm::slide_left(src, mask, lhs, rhs, count); }
{
    return xmm::slide_left(src, mask, lhs, rhs, count);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_left(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, vector<E> rhs, size_t count) noexcept
requires requires { xmm::slide_left(zero, mask, lhs, rhs, count); }
{
    return xmm::slide_left(zero, mask, lhs, rhs, count);
}

template <simd_element E, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_right(
    abi_tag, vector<E> lhs, vector<E> rhs, N count) noexcept
requires requires { xmm::slide_left<(vector<E>::size() - N::value)>(lhs, rhs); }
{
    static_assert(N::value <= vector<E>::size());
    return xmm::slide_left<(vector<E>::size() - N::value)>(lhs, rhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_right(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs, N) noexcept
requires requires {
    xmm::slide_right(src, mask, lhs, rhs, imm<vector<E>::size() - N::value>);
}
{
    static_assert(N::value <= vector<E>::size());
    return xmm::slide_right(
        src, mask, lhs, rhs, imm<vector<E>::size() - N::value>);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_right(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, vector<E> rhs, N) noexcept
requires requires {
    xmm::slide_right(zero, mask, lhs, rhs, imm<vector<E>::size() - N::value>);
}
{
    static_assert(N::value <= vector<E>::size());
    return xmm::slide_right(
        zero, mask, lhs, rhs, imm<vector<E>::size() - N::value>);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_right(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs, size_t count) noexcept
requires requires {
    xmm::slide_right(src, mask, lhs, rhs, vector<E>::size() - count);
}
{
    return xmm::slide_right(src, mask, lhs, rhs, vector<E>::size() - count);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> slide_right(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, vector<E> rhs, size_t count) noexcept
requires requires {
    xmm::slide_right(zero, mask, lhs, rhs, vector<E>::size() - count);
}
{
    return xmm::slide_right(zero, mask, lhs, rhs, vector<E>::size() - count);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif // if DPL_SIMD_X86_SSE4_2

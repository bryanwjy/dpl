// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/arithmetic.h"
#  include "dpl/xmm/operations/bit/popcount.h"
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_zero(vector<E> val) noexcept {
    if constexpr (!unsigned_integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::countr_zero(xmm::reinterpret<bit>(val));
    } else {
        auto const nval = xmm::subtract(vector<E>(_mm_setzero_si128()), val);
        return xmm::popcount(xmm::subtract(
            vector<E>(_mm_and_si128(+val, +nval)), xmm::broadcast<E>(1u)));
    }
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_zero(vector<unsigned_representation_t<E>> src,
        cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::popcount(src, mask, src); }
{
    if constexpr (!unsigned_integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::countr_zero(src, mask, xmm::reinterpret<bit>(val));
    } else {
        auto const nval = xmm::subtract(vector<E>(_mm_setzero_si128()), val);
        return xmm::popcount(src, mask,
            xmm::subtract(
                vector<E>(_mm_and_si128(+val, +nval)), xmm::broadcast<E>(1u)));
    }
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_zero(
        dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires(vector<unsigned_representation_t<E>> src) {
    xmm::popcount(zero, mask, src);
}
{
    if constexpr (!unsigned_integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::countr_zero(zero, mask, xmm::reinterpret<bit>(val));
    } else {
        auto const nval = xmm::subtract(vector<E>(_mm_setzero_si128()), val);
        return xmm::popcount(zero, mask,
            xmm::subtract(
                vector<E>(_mm_and_si128(+val, +nval)), xmm::broadcast<E>(1u)));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_one(vector<E> val) noexcept {
    using bit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<bit>(val);
    return xmm::countr_zero(
        vector<bit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_one(vector<unsigned_representation_t<E>> src,
        cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countr_zero(src, mask, src); }
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countr_zero(src, mask,
        vector<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_one(
        dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires(vector<unsigned_representation_t<E>> rep) {
    xmm::countr_zero(zero, mask, rep);
}
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countr_zero(zero, mask,
        vector<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_zero(abi_tag, vector<E> val) noexcept
requires requires { xmm::countr_zero(val); }
{
    return xmm::countr_zero(val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_zero(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countr_zero(src, mask, val); }
{
    return xmm::countr_zero(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_zero(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countr_zero(zero, mask, val); }
{
    return xmm::countr_zero(zero, mask, val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_one(abi_tag, vector<E> val) noexcept
requires requires { xmm::countr_one(val); }
{
    return xmm::countr_one(val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_one(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countr_one(src, mask, val); }
{
    return xmm::countr_one(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_one(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countr_one(zero, mask, val); }
{
    return xmm::countr_one(zero, mask, val);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

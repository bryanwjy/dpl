// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
namespace internal {
[[maybe_unused]] alignas(16) inline constexpr char popcnt_lut[]{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
} // namespace internal

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline size_vector_t<E>
    DPL_VECTORCALL popcount(vector<E> val) noexcept {
    if constexpr (sizeof(E) == 1) {
#  if DPL_SIMD_X86_AVX512BITALG && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi8(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#  else
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(internal::popcnt_lut));
        auto const mask = _mm_set1_epi8(0x0f);
        auto const vval = +xmm::reinterpret<int8>(val);
        auto const lo = _mm_and_si128(vval, mask);
        auto const hi = _mm_and_si128(_mm_srli_epi16(vval, 4), mask);
        return _mm_add_epi8(
            _mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
#  endif
    } else if constexpr (sizeof(E) == 2) {
#  if DPL_SIMD_X86_AVX512BITALG && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi16(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#  else
        return _mm_maddubs_epi16(+xmm::popcount(xmm::reinterpret<uint8>(val)),
            +xmm::broadcast<int16>(0x0101));
#  endif
    } else if constexpr (sizeof(E) == 4) {
#  if DPL_SIMD_X86_AVX512VPOPCNTDQ && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi32(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#  else
        return _mm_madd_epi16(+xmm::popcount(xmm::reinterpret<uint16>(val)),
            +xmm::broadcast<int16>(1));
#  endif
    } else {
        static_assert(sizeof(E) == 8);
#  if DPL_SIMD_X86_AVX512VPOPCNTDQ && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi64(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#  else
        return _mm_sad_epu8(
            +xmm::popcount(xmm::reinterpret<uint8>(val)), _mm_setzero_si128());
#  endif
    }
}

#  if DPL_SIMD_X86_AVX512VL
#    if DPL_SIMD_X86_AVX512VPOPCNTDQ
template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL popcount(
        vector<uint32> src, cmask_t<uint32, M>, vector<E> val) noexcept {
    return _mm_mask_popcnt_epi32(+src, M, +xmm::reinterpret<uint32>(val));
}

template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint32, M>, vector<E> val) noexcept {
    return _mm_maskz_popcnt_epi32(M, +xmm::reinterpret<uint32>(val));
}

template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL popcount(
        vector<uint64> src, cmask_t<uint64, M>, vector<E> val) noexcept {
    return _mm_mask_popcnt_epi64(+src, M, +xmm::reinterpret<uint64>(val));
}

template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint64, M>, vector<E> val) noexcept {
    return _mm_maskz_popcnt_epi64(M, +xmm::reinterpret<uint64>(val));
}
#    endif

#    if DPL_SIMD_X86_AVX512BITALG
template <imask_t<uint16> M, common_size_with<uint16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint16>
    DPL_VECTORCALL popcount(
        vector<uint16> src, cmask_t<uint16, M>, vector<E> val) noexcept {
    return _mm_mask_popcnt_epi16(+src, M, +xmm::reinterpret<uint16>(val));
}

template <imask_t<uint16> M, common_size_with<uint16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint16>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint16, M>, vector<E> val) noexcept {
    return _mm_maskz_popcnt_epi16(M, +xmm::reinterpret<uint16>(val));
}

template <imask_t<uint8> M, common_size_with<uint8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint8>
    DPL_VECTORCALL popcount(
        vector<uint8> src, cmask_t<uint8, M>, vector<E> val) noexcept {
    return _mm_mask_popcnt_epi8(+src, M, +xmm::reinterpret<uint8>(val));
}

template <imask_t<uint8> M, common_size_with<uint8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint8>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint8, M>, vector<E> val) noexcept {
    return _mm_maskz_popcnt_epi8(M, +xmm::reinterpret<uint8>(val));
}
#    endif
#  endif

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline size_vector_t<E>
    DPL_VECTORCALL popcount(abi_tag, vector<E> val) noexcept
requires requires { xmm::popcount(val); }
{
    return xmm::popcount(val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline size_vector_t<E>
    DPL_VECTORCALL popcount(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::popcount(src, mask, val); }
{
    return xmm::popcount(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline size_vector_t<E>
    DPL_VECTORCALL popcount(
        abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::popcount(zero, mask, val); }
{
    return xmm::popcount(zero, mask, val);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

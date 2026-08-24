// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/load.h"
#    include "dpl/xmm/basic/store.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <simd_element E>
requires unsigned_integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL abs(vector<E> val) noexcept {
    return val;
}

template <simd_element E>
requires signed_integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL abs(vector<E> val) noexcept {
    if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        return _mm_abs_epi64(+val);
#  else
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto const sign = _mm_cmpgt_epi64(zero, vval);
        return _mm_sub_epi64(_mm_xor_si128(vval, sign), sign);
#  endif
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_abs_epi32(+val);
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_abs_epi16(+val);
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        return _mm_abs_epi8(+val);
    }
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr vector<float> DPL_VECTORCALL abs(vector<float> val) noexcept {
    if consteval {
        alignas(16) uint32 data[4]{};
        xmm::store(xmm::reinterpret<uint32>(val), data);
        for (auto& lane : data) {
            constexpr auto mask = ~__DPL bit_cast<uint32>(-0.0f);
            lane &= mask;
        }
        return xmm::reinterpret<float>(xmm::load<uint32>(data));
    } else {
        return _mm_andnot_ps(_mm_set1_ps(-0.0f), +val);
    }
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr vector<double> DPL_VECTORCALL abs(vector<double> val) noexcept {
    if consteval {
        alignas(16) uint64 data[2]{};
        xmm::store(xmm::reinterpret<uint64>(val), data);
        for (auto& lane : data) {
            constexpr auto mask = ~__DPL bit_cast<uint64>(-0.0);
            lane &= mask;
        }
        return xmm::reinterpret<double>(xmm::load<uint64>(data));
    } else {
        return _mm_andnot_pd(_mm_set1_pd(-0.0), +val);
    }
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr vector<ext::float16> DPL_VECTORCALL abs(
    vector<ext::float16> val) noexcept {
    if consteval {
        alignas(16) uint16 data[8]{};
        xmm::store(xmm::reinterpret<uint16>(val), data);
        for (auto& lane : data) {
            constexpr auto mask = ~static_cast<uint16>(0x8000u);
            lane &= mask;
        }
        return xmm::reinterpret<ext::float16>(xmm::load<uint16>(data));
    } else {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        return _mm_abs_ph(+val);
#  else
        auto const vval = +xmm::reinterpret<int16>(val);
        return __DPL bit_cast<__m128h>(
            _mm_andnot_si128(_mm_set1_epi16(0x8000), vval));
#  endif
    }
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr vector<ext::bfloat16> DPL_VECTORCALL abs(
    vector<ext::bfloat16> val) noexcept {
    if consteval {
        alignas(16) uint16 data[8]{};
        xmm::store(xmm::reinterpret<uint16>(val), data);
        for (auto& lane : data) {
            constexpr auto mask = ~static_cast<uint16>(0x8000u);
            lane &= mask;
        }
        return xmm::reinterpret<ext::bfloat16>(xmm::load<uint16>(data));
    } else {
        auto const vval = +xmm::reinterpret<int16>(val);
        return __DPL bit_cast<__m128bh>(
            _mm_andnot_si128(_mm_set1_epi16(0x8000), vval));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> abs(abi_tag, vector<E> val) noexcept
requires requires { xmm::abs(val); }
{
    return xmm::abs(val);
}

#  if DPL_SIMD_X86_AVX512VL
#    if DPL_SIMD_X86_AVX512F
template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL abs(
        vector<int32> src, cmask_t<int32, M> mask, vector<int32> val) noexcept {
    return _mm_mask_abs_epi32(+src, M, +val);
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL abs(
        vector<int64> src, cmask_t<int64, M> mask, vector<int64> val) noexcept {
    return _mm_mask_abs_epi64(+src, M, +val);
}

template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int32, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_abs_epi32(M, +val);
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int64, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_abs_epi64(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512BW
template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8>
    DPL_VECTORCALL abs(
        vector<int8> src, cmask_t<int8, M> mask, vector<int8> val) noexcept {
    return _mm_mask_abs_epi32(+src, M, +val);
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16>
    DPL_VECTORCALL abs(
        vector<int16> src, cmask_t<int16, M> mask, vector<int16> val) noexcept {
    return _mm_mask_abs_epi64(+src, M, +val);
}

template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int8, M> mask, vector<int8> val) noexcept {
    return _mm_maskz_abs_epi32(M, +val);
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int16, M> mask, vector<int16> val) noexcept {
    return _mm_maskz_abs_epi64(M, +val);
}
#    endif
#  endif

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL abs(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::abs(src, mask, val); }
{
    return xmm::abs(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL abs(
        abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::abs(zero, mask, val); }
{
    return xmm::abs(zero, mask, val);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

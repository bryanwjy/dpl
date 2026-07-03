// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/load.h"
#  include "dpl/xmm/basic/store.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element E>
requires unsigned_integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL abs(simd<E> val) noexcept {
    return val;
}

DPL_EXPORT template <simd_element E>
requires signed_integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL abs(simd<E> val) noexcept {
    if constexpr (sizeof(E) == sizeof(int64)) {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        return _mm_abs_epi64(+val);
#else
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto const sign = _mm_cmpgt_epi64(zero, vval);
        return _mm_sub_epi64(_mm_xor_si128(vval, sign), sign);
#endif
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_abs_epi32(+val);
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_abs_epi16(+val);
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        return _mm_abs_epi8(+val);
    }
}

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr simd<float>
    DPL_VECTORCALL abs(simd<float> val) noexcept {
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

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr simd<double>
    DPL_VECTORCALL abs(simd<double> val) noexcept {
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

DPL_EXPORT template <simd_element E>
requires bfloat16_like<E> || float16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr simd<E>
    DPL_VECTORCALL abs(simd<E> val) noexcept {
    if consteval {
        alignas(16) uint16 data[8]{};
        xmm::store(xmm::reinterpret<uint16>(val), data);
        for (auto& lane : data) {
            constexpr auto mask = ~static_cast<uint16>(0x8000u);
            lane &= mask;
        }
        return xmm::reinterpret<E>(xmm::load<uint16>(data));
    } else {
        if constexpr (bfloat16_like<E>) {
            auto const vval = +xmm::reinterpret<int16>(val);
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_andnot_si128(_mm_set1_epi16(0x8000), vval));
        } else {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
            return _mm_abs_ph(+val);
#else
            auto const vval = +xmm::reinterpret<int16>(val);
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_andnot_si128(_mm_set1_epi16(0x8000), vval));
#endif
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr simd<E>
    DPL_VECTORCALL abs(abi_tag, simd<E> val) noexcept
requires requires { xmm::abs(val); }
{
    return xmm::abs(val);
}

#if DPL_SIMD_X86_AVX512VL
#  if DPL_SIMD_X86_AVX512F
DPL_EXPORT template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int32>
    DPL_VECTORCALL abs(
        simd<int32> src, cmask_t<int32, M> mask, simd<int32> val) noexcept {
    return _mm_mask_abs_epi32(+src, M, +val);
}

DPL_EXPORT template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int64>
    DPL_VECTORCALL abs(
        simd<int64> src, cmask_t<int64, M> mask, simd<int64> val) noexcept {
    return _mm_mask_abs_epi64(+src, M, +val);
}

DPL_EXPORT template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int32>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int32, M> mask, simd<int32> val) noexcept {
    return _mm_maskz_abs_epi32(M, +val);
}

DPL_EXPORT template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int64>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int64, M> mask, simd<int64> val) noexcept {
    return _mm_maskz_abs_epi64(M, +val);
}
#  endif

#  if DPL_SIMD_X86_AVX512BW
DPL_EXPORT template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int8>
    DPL_VECTORCALL abs(
        simd<int8> src, cmask_t<int8, M> mask, simd<int8> val) noexcept {
    return _mm_mask_abs_epi32(+src, M, +val);
}

DPL_EXPORT template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int16>
    DPL_VECTORCALL abs(
        simd<int16> src, cmask_t<int16, M> mask, simd<int16> val) noexcept {
    return _mm_mask_abs_epi64(+src, M, +val);
}

DPL_EXPORT template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int8>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int8, M> mask, simd<int8> val) noexcept {
    return _mm_maskz_abs_epi32(M, +val);
}

DPL_EXPORT template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<int16>
    DPL_VECTORCALL abs(
        dx::zero_t, cmask_t<int16, M> mask, simd<int16> val) noexcept {
    return _mm_maskz_abs_epi64(M, +val);
}
#  endif
#endif

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL abs(abi_tag, type_identity_t<simd<E>> src,
        type_identity_t<mask<E>> mask, simd<E> val) noexcept
requires requires { xmm::abs(src, mask, val); }
{
    return xmm::abs(src, mask, val);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL abs(abi_tag, dx::zero_t zero, type_identity_t<mask<E>> mask,
        simd<E> val) noexcept
requires requires { xmm::abs(zero, mask, val); }
{
    return xmm::abs(zero, mask, val);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

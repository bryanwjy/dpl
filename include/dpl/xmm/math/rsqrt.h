// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/xmm/math/fmadd.h"
#include "dpl/xmm/math/fnmsub.h"
#include "dpl/xmm/math/product_error.h"

#if !DPL_MODULES
#  include "dpl/core/math/accuracy.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/operations/compare.h"

#  include <immintrin.h>
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL rsqrt(vector<float> val,
    accuracy::approximate_t = accuracy::approximate) noexcept {
    // Should this handle flush-to-zero ftz behaviour?
    return _mm_rsqrt14_ps(+val); // flushes subnormal results to zero
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL rsqrt(vector<float> src, cmask_t<float, M> mask,
        vector<float> val, accuracy::approximate_t approx) noexcept {
    if constexpr (all_of(mask)) {
        return xmm::rsqrt(val, approx);
    } else if constexpr (none_of(mask)) {
        return src;
    } else {
        return _mm_mask_rsqrt14_ps(+src, M, +val);
    }
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL rsqrt(dx::zero_t zero, cmask_t<float, M> mask,
        vector<float> val, accuracy::approximate_t approx) noexcept {
    if constexpr (all_of(mask)) {
        return xmm::rsqrt(val, approx);
    } else if constexpr (none_of(mask)) {
        return xmm::broadcast<float>(zero);
    } else {
        return _mm_maskz_rsqrt14_ps(M, +val);
    }
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL rsqrt(
    vector<float> val, accuracy::precise_t) noexcept {
    constexpr auto post_norm = 0x1.0p-32f; // sqrt so halved
    constexpr auto pre_norm = 0x1.0p-64f;
    auto const vone = xmm::broadcast<float>(1.0f);
    auto const vhalf = xmm::broadcast<float>(0.5f);
    auto const v3_8 = xmm::broadcast<float>(0x3.0p-3f);
    auto const islarge =
        _mm_cmp_ps_mask(+val, +xmm::broadcast<float>(0x1.0p126f), _CMP_GE_OQ);

    {
        auto const vpre_norm = xmm::broadcast<float>(pre_norm);
        val = _mm_mask_mul_ps(+val, islarge, +vpre_norm, +val);
    }

    auto const y = xmm::rsqrt(val, accuracy::approximate);
    auto const xyhi = xmm::multiply(val, y);
    auto const xylo = details::product_error(val, y, xyhi);
    auto residual = xmm::fnmsub(xyhi, y, vone);
    residual = xmm::fnmsub(xylo, y, residual);
    auto result = xmm::fmadd(residual, v3_8, vhalf);
    result = xmm::fmadd(xmm::multiply(y, residual), result, y);
    auto const vpost_norm = xmm::broadcast<float>(post_norm);
    result = _mm_mask_mul_ps(+result, islarge, +vpost_norm, +result);

    auto const isspecial =
        _mm_fpclass_ps_mask(+val, 0x01 | 0x02 | 0x04 | 0x08 | 0x80);
    // QNaN, +0, -0, +inf, SNaN
    return _mm_mask_mov_ps(+result, isspecial, +y);
}

#elif DPL_SIMD_X86_SSE2

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL rsqrt(
    vector<float> val, accuracy::approximate_t) noexcept {
    constexpr auto post_norm = 0x1.0p32f; // sqrt so halved
    constexpr auto pre_norm = 0x1.0p64f;
    auto const vone = xmm::broadcast<float>(1.0f);
    auto const vhalf = xmm::broadcast<float>(0.5f);
    auto const issmall = [](vector<float> val) {
        auto const vlower_bound = xmm::broadcast<float>(dx::min_value);
        return xmm::cmplt(val, vlower_bound);
    }(val);

    {
        auto const vpre_norm = xmm::broadcast<float>(pre_norm);
        auto const scale = xmm::select(issmall, vpre_norm, vone);
        val = xmm::multiply(scale, val);
    }

    // Should this also handle result flush-to-zero ftz behaviour?
    vector<float> y = _mm_rsqrt_ps(+val); // flushes subnormal results to zero

    auto result = xmm::multiply(val, y);
#  if DPL_SIMD_X86_FMA
    // The scope is just to fix sublime's highlighting
    {
        result = xmm::fnmsub(result, y, vone);
        result = xmm::fmadd(xmm::multiply(y, vhalf), result, y);
    }
#  else
    {
        result = xmm::subtract(vone, xmm::multiply(result, y));
        result = xmm::multiply(xmm::multiply(y, vhalf), result);
        result = xmm::add(result, y);
    }
#  endif
    {
        auto const vpost_norm = xmm::broadcast<float>(post_norm);
        auto const scale = xmm::select(issmall, vpost_norm, vone);
        result = xmm::multiply(scale, result);
    }

    return result;
}

#  if DPL_SIMD_X86_FMA
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL rsqrt(
    vector<float> val, accuracy::precise_t) noexcept {
    constexpr auto post_norm = 0x1.0p-32f; // sqrt so halved
    constexpr auto pre_norm = 0x1.0p-64f;
    auto const vone = xmm::broadcast<float>(1.0f);
    auto const vhalf = xmm::broadcast<float>(0.5f);
    auto const v3_8 = xmm::broadcast<float>(0x3.0p-3f);
    auto const islarge = [](vector<float> val) {
        auto const vupper_bound = xmm::broadcast<float>(0x1.0p126f);
        return xmm::cmpge(val, vupper_bound);
    }(val);
    auto const issmall = [](vector<float> val) {
        auto const vlower_bound = xmm::broadcast<float>(dx::min_value);
        return xmm::cmplt(val, vlower_bound);
    }(val);

    {
        constexpr auto inv_pre_norm = 1.0f / pre_norm;
        auto const vpre_norm = xmm::broadcast<float>(pre_norm);
        auto const vipre_norm = xmm::broadcast<float>(inv_pre_norm);

        auto scale = xmm::select(islarge, vpre_norm, vone);
        scale = xmm::select(issmall, vipre_norm, scale);
        val = xmm::multiply(scale, val);
    }

    vector<float> y = _mm_rsqrt_ps(+val);

    auto const xyhi = xmm::multiply(val, y);
    auto const xylo = details::product_error(val, y, xyhi);

    auto residual = dx::fnmsub(xyhi, y, vone);
    residual = dx::fnmsub(xylo, y, residual);
    auto result = xmm::fmadd(residual, v3_8, vhalf);
    result = xmm::fmadd(xmm::multiply(y, residual), result, y);

    auto const vpost_norm = xmm::broadcast<float>(post_norm);
    constexpr auto inv_post_norm = 1.0f / post_norm;
    auto const vipost_norm = xmm::broadcast<float>(inv_post_norm);

    auto scale = xmm::select(islarge, vpost_norm, vone);
    scale = xmm::select(issmall, vipost_norm, scale);
    result = xmm::multiply(scale, result);

    return result;
}
#  else
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL rsqrt(
    vector<float> val, accuracy::precise_t) noexcept {
    auto const vone = xmm::broadcast<float>(1.0f);
    auto const vhalf = xmm::broadcast<float>(0.5f);
    auto const v3_8 = xmm::broadcast<float>(0x3.0p-3f);

    vector<float> y = _mm_div_ps(vone, _mm_sqrt_ps(+val));

    auto const islarge = xmm::cmpge(val, xmm::broadcast<float>(0x1.0p103f));
    auto const xs = xmm::multiply(
        val, xmm::select(islarge, xmm::broadcast<float>(0x1.0p-64f), vone));
    auto const ys = xmm::multiply(
        y, xmm::select(islarge, xmm::broadcast<float>(0x1.0p32f), vone));

    auto const xyhi = xmm::multiply(xs, ys);
    auto const xylo = details::product_error(xs, ys, xyhi);

    auto const xy2hi = xmm::multiply(xyhi, ys);
    auto const xy2lo = details::product_error(xyhi, ys, xy2hi);
    auto residual = xmm::subtract(xmm::subtract(vone, xy2hi), xy2lo);
    residual = xmm::subtract(residual, xmm::multiply(xylo, ys));
    auto result = xmm::multiply(ys, vhalf);
    result = xmm::multiply(result, residual);
    result = xmm::add(result, ys);
    result = xmm::multiply(
        result, xmm::select(islarge, xmm::broadcast<float>(0x1.0p-32f), vone));

    auto const valid = xmm::bwand( //
        xmm::cmpgt(val, dx::zero),
        xmm::cmplt(val, dx::infinity)); // false for NaN
    return xmm::select(valid, result, y);
}
#  endif

#endif // DPL_SIMD_X86_SSE2

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> rsqrt(
    abi_tag, vector<E> val, accuracy::approximate_t approximate) noexcept
requires requires { xmm::rsqrt(val, approximate); }
{
    return xmm::rsqrt(val, approximate);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> rsqrt(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val,
    accuracy::approximate_t approximate) noexcept
requires requires { xmm::rsqrt(src, mask, val, approximate); }
{
    return xmm::rsqrt(src, mask, val, approximate);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> rsqrt(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> val, accuracy::approximate_t approximate) noexcept
requires requires { xmm::rsqrt(zero, mask, val, approximate); }
{
    return xmm::rsqrt(zero, mask, val, approximate);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> rsqrt(
    abi_tag, vector<E> val, accuracy::precise_t precise) noexcept
requires requires { xmm::rsqrt(val, precise); }
{
    return xmm::rsqrt(val, precise);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> rsqrt(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val, accuracy::precise_t precise) noexcept
requires requires { xmm::rsqrt(src, mask, val, precise); }
{
    return xmm::rsqrt(src, mask, val, precise);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> rsqrt(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> val, accuracy::precise_t precise) noexcept
requires requires { xmm::rsqrt(zero, mask, val, precise); }
{
    return xmm::rsqrt(zero, mask, val, precise);
}

} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END

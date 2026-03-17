

// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/abs.h"
#include "dpl/xmm/operations/arithmetic.h"
#include "dpl/xmm/operations/bit.h"
#include "dpl/xmm/operations/bitwise.h"
#include "dpl/xmm/operations/select.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/broadcast.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <typename T, typename U>
concept integral_cast_target_like = integral<U> &&
    (common_integral_with<T, signed_representation_t<U>> ||
        common_integral_with<T, unsigned_representation_t<U>>);

DPL_EXPORT template <common_float_with<double> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <common_float_with<double> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtps_pd(+src);
}

DPL_EXPORT template <common_float_with<double> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtph_pd(+src);
#else
    using sint = signed_representation_t<To>;
    using uint = unsigned_representation_t<To>;
    auto const arg = xmm::cast<uint>(tag, xmm::reinterpret<uint16>(tag, src));
    auto const msb = xmm::broadcast<uint>(tag, 0x8000);
    auto const signs = xmm::reinterpret<To>(
        tag, xmm::bwshift_left<48>(tag, xmm::bwand(tag, arg, msb)));
    auto const parg = xmm::bwshift_left<42>(tag, xmm::bwandnot(tag, arg, msb));

    constexpr auto exp_mask =
        xmm::broadcast<uint>(tag, static_cast<uint>(0x1f) << 52);
    auto const isfinite = simd<uint>(_mm_cmpgt_epi64(+exp_mask, parg));

    constexpr auto inf64 = xmm::broadcast<uint>(
        tag, __DPL bit_cast<uint>(dx::infinity_v<double>));
    auto const nonfinite =
        xmm::bwandnot(tag, xmm::bwor(tag, inf64, parg), isfinite);
    constexpr auto shift = xmm::broadcast<To>(tag, 0x1p1008);
    auto const shifted =
        xmm::multiply(tag, xmm::reinterpret<To>(tag, parg), shift);
    auto const abs_f64 =
        xmm::reinterpret<To>(tag, xmm::bwor(tag, nonfinite, shifted));
    return xmm::bwor(tag, abs_f64, signs);
#endif
}

// Without AVX512, it is simply not worth it
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
DPL_EXPORT template <common_float_with<double> To,
    common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
    return _mm_cvtepi64_pd(+src);
}
#endif

DPL_EXPORT template <common_float_with<double> To,
    common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (signed_integral<E>) {
        return _mm_cvtepi32_pd(+src);
    } else {
        auto const large = _mm_cmplt_epi32(+src, _mm_setzero_si128());
        auto const islolarge =
            _mm_castsi128_pd(_mm_unpacklo_epi32(large, large));
        return _mm_add_pd(
            _mm_and_pd(islolarge, _mm_set1_pd(0x1p32)), _mm_cvtepi32_pd(+src));
    }
}

DPL_EXPORT template <common_float_with<double> To,
    common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepi32_pd(_mm_cvtepu16_epi32(+src));
    } else {
        return _mm_cvtepi32_pd(_mm_cvtepi16_epi32(+src));
    }
}

DPL_EXPORT template <common_float_with<double> To,
    common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        auto const lo =
            _mm_cvtepu16_epi32(_mm_unpacklo_epi8(+src, _mm_setzero_si128()));
        return _mm_cvtepi32_pd(lo);
    } else {
        auto const lo = _mm_cvtepi16_epi32(
            _mm_srai_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), +src), 8));
        return _mm_cvtepi32_pd(lo);
    }
}

DPL_EXPORT template <common_float_with<float> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtpd_ps(+src);
}

DPL_EXPORT template <common_float_with<float> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <common_float_with<float> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_F16C
    return _mm_cvtph_ps(+src);
#else
    using sint = signed_representation_t<To>;
    using uint = unsigned_representation_t<To>;
    auto const arg = xmm::cast<uint>(tag, xmm::reinterpret<uint16>(tag, src));
    auto const msb = xmm::broadcast<uint>(tag, 0x8000);
    auto const signs = xmm::reinterpret<To>(
        tag, xmm::bwshift_left<16>(tag, xmm::bwand(tag, arg, msb)));
    auto const parg = xmm::bwshift_left<13>(tag, xmm::bwandnot(tag, arg, msb));

    constexpr auto exp_mask = xmm::broadcast<uint>(tag, 0xf800000);
    auto const isnotfinite = simd<uint>(_mm_cmpge_epi32(parg, +exp_mask));

    constexpr auto inf32 = xmm::broadcast<uint>(tag, 0x7f800000);
    auto const nonfinite =
        xmm::bwand(tag, isnotfinite, xmm::bwor(tag, inf32, parg));
    constexpr auto shift = xmm::broadcast<To>(tag, 0x1p112f);
    auto const shifted =
        xmm::multiply(tag, xmm::reinterpret<To>(tag, parg), shift);

    auto const abs_f32 =
        xmm::reinterpret<To>(tag, xmm::bwor(tag, nonfinite, shifted));
    return xmm::bwor(tag, abs_f32, signs);
#endif
}

DPL_EXPORT template <common_float_with<float> TE, brain_float FE>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<TE> DPL_VECTORCALL cast(abi_tag, simd<FE> from) noexcept {
#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtpbh_ps(+from);
#else
    auto const src = __DPL bit_cast<__m128i>(+from);
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(src), 16));
#endif
}

DPL_EXPORT template <common_float_with<float> To,
    common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu64_ps(+src);
    } else {
        return _mm_cvtepi64_ps(+src);
    }
#else
    auto const zero = _mm_setzero_si128();
    if constexpr (signed_integral<E>) {
        auto const hisign = _mm_and_si128(_mm_set1_epi64x(1ll << 63), +src);
        auto const losign =
            _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(hisign), 32));

        src = xmm::abs(tag, src);
        auto const ishizero = _mm_cmpeq_epi32(+src, zero);
        auto const hicorrection = _mm_andnot_si128(
            ishizero, _mm_or_si128(_mm_set1_epi64x(32ll << 55), hisign));
        auto const islarge = _mm_castsi128_ps(_mm_cmplt_epi32(+src, zero));
        auto const locorrection = _mm_and_ps(islarge, _mm_set1_ps(0x1p32f));
        auto hilohilo = _mm_add_ps(
            _mm_castsi128_ps(_mm_add_epi64(
                _mm_castps_si128(_mm_cvtepi32_ps(+src)), hicorrection)),
            locorrection);
        hilohilo = _mm_xor_ps(hilohilo, losign);
        auto const hihilolo = _mm_castps_si128(
            _mm_shuffle_ps(hilohilo, hilohilo, _MM_SHUFFLE(3, 1, 2, 0)));
        return _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi64(hihilolo, zero)),
            _mm_castsi128_ps(_mm_unpackhi_epi64(hihilolo, zero)));
    } else {
        auto const islarge = _mm_castsi128_ps(_mm_cmplt_epi32(+src, zero));
        auto const correction = _mm_and_ps(islarge, _mm_set1_ps(0x1p32f));
        auto const ishizero = _mm_cmpeq_epi32(+src, zero);
        auto const hilohilo = _mm_add_ps(
            _mm_castsi128_ps(
                _mm_add_epi64(_mm_castps_si128(_mm_cvtepi32_ps(+src)),
                    _mm_andnot_si128(ishizero, _mm_set1_epi64x(32ll << 55)))),
            correction);
        auto const hihilolo = _mm_castps_si128(
            _mm_shuffle_ps(hilohilo, hilohilo, _MM_SHUFFLE(3, 1, 2, 0)));

        return _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi64(hihilolo, zero)),
            _mm_castsi128_ps(_mm_unpackhi_epi64(hihilolo, zero)));
    }
#endif
}

DPL_EXPORT template <common_float_with<float> To,
    common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (signed_integral<E>) {
        return _mm_cvtepi32_ps(+src);
    } else {
        constexpr auto max = max_value_v<make_signed_t<E>>;
        auto const gt = _mm_cmpgt_epi32(+src, _mm_set1_epi32(max));
        return _mm_add_ps(
            _mm_cvtepi32_ps(+src), _mm_and_ps(gt, _mm_set1_ps(0x1p32f)));
    }
}

DPL_EXPORT template <common_float_with<float> To,
    common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepi32_ps(_mm_cvtepu16_epi32(+src));
    } else {
        return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(+src));
    }
}

DPL_EXPORT template <common_float_with<float> To,
    common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        auto const lo =
            _mm_cvtepu16_epi32(_mm_unpacklo_epi8(+src, _mm_setzero_si128()));
        return _mm_cvtepi32_ps(lo);
    } else {
        auto const lo = _mm_cvtepi16_epi32(
            _mm_srai_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), +src), 8));
        return _mm_cvtepi32_ps(lo);
    }
}

DPL_EXPORT template <brain_float TE, common_float_with<float> FE>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<TE> DPL_VECTORCALL cast(abi_tag, simd<FE> from) noexcept {
#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtneps_pbh(+from);
#else
    auto const ival = _mm_castps_si128(+from);
    // round to nearest even
    auto const lsb = _mm_and_si128(_mm_srli_epi32(ival, 16), _mm_set1_epi32(1));
    auto const bias = _mm_add_epi32(_mm_set1_epi32(0x7fff), lsb);
    return __DPL bit_cast<__m128bh>(
        _mm_srli_epi32(_mm_add_epi32(ival, bias), 16));
#endif
}

DPL_EXPORT template <brain_float To, arithmetic_type E>
requires (!common_float_with<E, float>) &&
    requires(abi_tag tag, simd<E> src) { xmm::cast<float>(tag, src); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<To>(tag, xmm::cast<float>(tag, src));
}

DPL_EXPORT template <arithmetic_type To, brain_float E>
requires (!common_float_with<To, float>) &&
    requires(abi_tag tag, simd<float> src) { xmm::cast<To>(tag, src); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<To>(tag, xmm::cast<float>(tag, src));
}

DPL_EXPORT template <fp16_like To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_F16C
    return __DPL bit_cast<__m128h>(
        _mm_cvtps_ph(+src, _MM_FROUND_TO_NEAREST_INT));
#else
    using sint = signed_representation_t<E>;
    using uint = unsigned_representation_t<E>;
    auto const i32 = xmm::reinterpret<sint>(tag, src);
    auto const msb32 = xmm::broadcast<sint>(tag, dx::msb);
    auto const sign16 = xmm::bwshift_right<16>(
        tag, xmm::reinterpret<uint>(xmm::bwand(tag, msb32, i32)));
    auto const abs = xmm::reinterpret<E>(tag, xmm::bwandnot(tag, i32, msb32));
    auto const shifted =
        xmm::multiply(tag, abs, xmm::broadcast<E>(tag, 0x1p-112f));

    auto const rounded = xmm::bwshift_right<13>(tag,
        xmm::add(tag,
            xmm::bwand(tag, xmm::bwshift_right<13>(tag, shifted),
                xmm::broadcast<uint>(tag, dx::one)),
            xmm::add(tag, xmm::reinterpret<uint>(tag, shifted),
                xmm::broadcast<uint>(tag, 0x1000u))));

    auto const limit = xmm::broadcast<E>(tag, 0x1p16f);
    auto const isinf = simd<E>(_mm_cmpge_ps(+abs, +limit));
    auto const isnan = _mm_cmpunord_ps(+src, +src);

    auto const inf16 = xmm::broadcast<To>(tag, dx::infinity);
    auto const f16 = xmm::select(
        isinf, xmm::reinterpret<E>(inf16), xmm::reinterpret<E>(rounded));
    auto const result = xmm::reinterpret<uint>(tag, xmm::bwor(tag, isnan, f16));
    return xmm::reinterpret<To>(
        tag, simd<uint>(_mm_packus_epi32(+result, _mm_setzero_si128())));
#endif
}

DPL_EXPORT template <fp16_like To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtpd_ph(+src);
#else
    using sint = signed_representation_t<E>;
    using uint = unsigned_representation_t<E>;
    auto const i64 = xmm::reinterpret<sint>(tag, src);
    auto const msb64 = xmm::broadcast<sint>(tag, dx::msb);
    auto const sign16 = xmm::bwshift_right<48>(
        tag, xmm::reinterpret<uint>(xmm::bwand(tag, msb64, i64)));
    auto const abs = xmm::reinterpret<E>(tag, xmm::bwandnot(tag, i64, msb64));
    auto const shifted =
        xmm::multiply(tag, abs, xmm::broadcast<E>(tag, 0x1p-1008));
    auto const rounded = xmm::bwshift_right<42>(tag,
        xmm::add(tag,
            xmm::bwand(tag, xmm::bwshift_right<42>(tag, shifted),
                xmm::broadcast<uint>(tag, dx::one)),
            xmm::add(tag, xmm::reinterpret<uint>(tag, shifted),
                xmm::broadcast<uint>(tag, 0x1ull << 41))));
    auto const limit = xmm::broadcast<E>(tag, 0x1p16);
    auto const isinf = simd<E>(_mm_cmpge_pd(+abs, +limit));
    auto const isnan = _mm_cmpunord_pd(+src, +src);

    auto const inf16 = xmm::broadcast<To>(tag, dx::infinity);
    auto const f16 = xmm::select(
        isinf, xmm::reinterpret<E>(inf16), xmm::reinterpret<E>(rounded));
    auto const result = xmm::reinterpret<uint>(tag, xmm::bwor(tag, isnan, f16));

    using u16 = signed_representation_t<To>;
    // The cast here is just to ditribute the bits into place
    return xmm::reinterpret<To>(tag, xmm::cast<u16>(tag, result));
#endif
}

namespace details {

template <fp16_like To, common_float_with<float> E>
constexpr simd<To> to_postive_inthalf(simd<E> f32) noexcept {
    using u32 = unsigned_representation_t<E>;
    auto const bits = xmm::reinterpret<u32>(xmm::abi, f32);
    auto const mantissa =
        xmm::bwand(xmm::abi, xmm::bwshift_right<13>(xmm::abi, bits),
            xmm::broadcast<u32>(xmm::abi, 0x3ff));
    auto const exp = _mm_subs_epu16(
        +xmm::bwshift_right<23>(xmm::abi, bits), _mm_set1_epi32(112));
    auto const bexp = simd<u32>(_mm_slli_epi32(exp, 10));
    return xmm::reinterpret<To>(
        xmm::select(xmm::abi, mask<E>(_mm_cmpge_ps(f32, _mm_set1_ps(0x1p16f))),
            xmm::broadcast<u32>(0x7c00), xmm::bwor(xmm::abi, bexp, mantissa)));
}
} // namespace details

DPL_EXPORT template <fp16_like To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu64_ph(+src);
    } else {
        return _mm_cvtepi64_ph(+src);
    }
#else
    auto const zero = _mm_setzero_si128();
    if constexpr (unsigned_integral<E>) {
        using u16 = unsigned_representation_t<To>;
        using s32 = make_signed_t<bit_type_t<sizeof(u16) * 2 * char_bit_v>>;
        auto const all = _mm_set1_epi64x(0xffff);
        auto const isinf = mask<u16>(_mm_packs_epi32(
            _mm_packs_epi32(_mm_cmpgt_epi64(+src, all), zero), zero));
        auto const trunc32 =
            simd<s32>(_mm_packus_epi32(_mm_and_si128(+src, all), zero));
        auto const trunc =
            details::to_postive_inthalf<To>(xmm::cast<float>(tag, trunc32));
        return xmm::select(
            tag, isinf, xmm::broadcast<To>(tag, dx::infinity), trunc);
    } else {
        using u64 = make_unsigned_t<E>;
        using u16 = unsigned_representation_t<To>;
        auto const abs = xmm::reinterpret<u64>(tag, xmm::abs(tag, src));
        auto const vsign16 = simd<u16>(
            _mm_and_si128(_mm_packs_epi32(_mm_packs_epi32(+src, zero), zero),
                _mm_set1_epi16(0x8000)));

        return xmm::reinterpret<To>(tag,
            xmm::bit_fill(tag, vsign16,
                xmm::cast<To>(tag, xmm::reinterpret<u64>(tag, abs))));
    }
#endif
}

DPL_EXPORT template <fp16_like To, common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu32_ph(+src);
    } else {
        return _mm_cvtepi32_ph(+src);
    }
#else
    auto const zero = _mm_setzero_si128();
    using u16 = unsigned_representation_t<To>;
    if constexpr (unsigned_integral<E>) {
        auto const all = _mm_set1_epi32(0xffff);
        auto const isinf =
            mask<u16>(_mm_packs_epi32(_mm_cmpgt_epi32(+src, all), zero));
        auto const clamped = simd<E>(_mm_and_si128(+src, all));
        auto const trunc =
            details::to_postive_inthalf<To>(xmm::cast<float>(tag, clamped));
        return xmm::select(
            tag, isinf, xmm::broadcast<To>(tag, dx::infinity), trunc);
    } else {
        using u32 = unsigned_representation_t<E>;
        auto const abs = xmm::reinterpret<u32>(xmm::abs(tag, src));
        auto const vsign16 = simd<u16>(
            _mm_and_si128(_mm_packs_epi32(+src, zero), _mm_set1_epi16(0x8000)));
        return xmm::reinterpret<To>(tag,
            xmm::bit_fill(tag, vsign16,
                xmm::cast<To>(tag, xmm::reinterpret<u32>(tag, abs))));
    }
#endif
}

DPL_EXPORT template <fp16_like To, common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu16_ph(+src);
    } else {
        return _mm_cvtepi16_ph(+src);
    }
#else
    if constexpr (unsigned_integral<E>) {
        auto const zero = _mm_setzero_si128();
        using u32 = bit_type_t<sizeof(E) * 2 * char_bit_v>;
        auto const lo =
            xmm::cast<float>(tag, simd<u32>(_mm_unpacklo_epi16(+src, zero)));
        auto const hi =
            xmm::cast<float>(tag, simd<u32>(_mm_unpackhi_epi16(+src, zero)));
        auto const lo16 =
            xmm::reinterpret<float>(tag, details::to_postive_inthalf<To>(lo));
        auto const hi16 =
            xmm::reinterpret<float>(tag, details::to_postive_inthalf<To>(hi));
        using simdf = remove_const_t<decltype(lo)>;
        return xmm::reinterpret<To>(
            tag, simdf(_mm_shuffle_ps(lo16, hi16, _MM_SHUFFLE(1, 0, 1, 0))));
    } else {
        using u32 = bit_type_t<sizeof(E) * 2 * char_bit_v>;
        using u16 = __DPL make_unsigned_t<E>;
        auto const sign = xmm::reinterpret<To>(
            simd<u32>(_mm_and_si128(+src, _mm_set1_epi16(0x8000))));
        auto const abs =
            xmm::cast<To>(tag, xmm::reinterpret<u16>(xmm::abs(tag, src)));
        return xmm::bwor(tag, sign, abs);
    }
#endif
}

DPL_EXPORT template <fp16_like To, common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    using i16 = signed_representation_t<To>;
    return _mm_cvtepi16_ph(+xmm::cast<i16>(tag, src));
#else
    if constexpr (unsigned_integral<E>) {
        return xmm::cast<To>(tag, xmm::cast<uint16>(tag, src));
    } else {
        return xmm::cast<To>(tag, xmm::cast<int16>(tag, src));
    }
#endif
}

DPL_EXPORT template <integral_cast_target_like<int64> To,
    common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<To>) {
        return _mm_cvttpd_epu64(+src);
    } else {
        return _mm_cvttpd_epi64(+src);
    }
#else
    return _mm_set_epi64x(
        static_cast<To>(src[imm<1>]), static_cast<To>(src[imm<0>]));
#endif
}

DPL_EXPORT template <integral_cast_target_like<int64> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<To>) {
        return _mm_cvttph_epu64(+src);
    } else {
        return _mm_cvttph_epi64(+src);
    }
#else
    return xmm::cast<To>(tag, xmm::cast<float>(tag, src));
#endif
}

DPL_EXPORT template <integral_cast_target_like<int64> To,
    common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<To>) {
        return _mm_cvttps_epu64(+src);
    } else {
        return _mm_cvttps_epi64(+src);
    }
#else
    return _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(static_cast<int64>(src[imm<0>])),
        _mm_cvtsi64_si128(static_cast<int64>(src[imm<1>])));
#endif
}

DPL_EXPORT template <integral_cast_target_like<int64> To,
    common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <integral_cast_target_like<int64> To,
    common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu32_epi64(+src);
    } else {
        return _mm_cvtepi32_epi64(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int64> To,
    common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu16_epi64(+src);
    } else {
        return _mm_cvtepi16_epi64(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int64> To,
    common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu8_epi64(+src);
    } else {
        return _mm_cvtepi8_epi64(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int32> To,
    common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<To>) {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        return _mm_cvttps_epu32(+src);
#else
        constexpr int32 magic = 0x4f000000;
        auto xmm0 = +src;
        auto xmm2 = _mm_castsi128_ps(_mm_set1_epi32(magic));
        auto xmm1 = _mm_cmple_ps(xmm2, xmm0);
        xmm2 = _mm_and_ps(xmm1, xmm2);
        xmm1 = _mm_castps_si128(_mm_slli_epi32(_mm_castps_si128(xmm1), 31));
        xmm0 = _mm_sub_ps(xmm0, xmm2);
        xmm0 = _mm_cvttps_epi32(xmm0);
        return _mm_xor_ps(xmm0, xmm1);
#endif
    } else {
        return _mm_cvttps_epi32(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int32> To,
    common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<To>) {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        return _mm_cvttpd_epu32(+src);
#else
        return _mm_set_epi32(0, 0, static_cast<uint32>(src[imm<1>]),
            static_cast<uint32>(src[imm<0>]));
#endif
    } else {
        return _mm_cvttpd_epi32(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int32> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    if constexpr (signed_integral<To>) {
        return _mm_cvttph_epi32(+src);
    } else {
        return _mm_cvttph_epu32(+src);
    }
#else
    if constexpr (signed_integral<To>) {
        // Cheaper to convert to float first
        return _mm_cvttps_epi32(+xmm::cast<float>(src));
    } else {
        constexpr auto mantissa_width = 23u;
        constexpr auto hidden_bit = 1u << mantissa_width;
        constexpr auto bias = 127;
        // There's likely a faster way, but this is just easier :p
        auto const f32 = +xmm::cast<float>(src);
        auto const u32 = _mm_castps_si128(f32);
        auto const invalid = _mm_or_ps(_mm_cmplt_epi32(u32, _mm_setzero_ps()),
            _mm_cmpge_epi32(u32, _mm_set1_epi32(0x7f800000)));
        auto const result = _mm_cvttps_epi32(f32);
        return _mm_castps_si128(
            _mm_blendv_ps(result, _mm_set1_ps(-0.0f), invalid));
    }
#endif
}

DPL_EXPORT template <integral_cast_target_like<int32> To,
    common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi32(+src);
#else
    return _mm_insert_epi64(
        _mm_shuffle_epi32(+src, _MM_SHUFFLE(2, 0, 2, 0)), 0, 1);
#endif
}

DPL_EXPORT template <integral_cast_target_like<int32> To,
    common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <integral_cast_target_like<int32> To,
    common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu16_epi32(+src);
    } else {
        return _mm_cvtepi16_epi32(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int32> To,
    common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu8_epi32(+src);
    } else {
        return _mm_cvtepi8_epi32(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int16> To, floating_point E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<To>(tag, xmm::cast<int32>(tag, src));
}

DPL_EXPORT template <integral_cast_target_like<int16> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL
    cast(abi_tag tag [[maybe_unused]], simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<To>) {
        auto const error = xmm::broadcast<E>(dx::msb);
        auto const neg = _mm_cmp_ph(+src, xmm::broadcast<E>(0), 17);
        return _mm_mask_blend_ph(neg, _mm_cvttph_epi16(+src), error);
    } else {
        return _mm_cvttph_epi16(+src);
    }
#else
    return xmm::cast<To>(tag, xmm::cast<int32>(tag, src));
#endif
}

DPL_EXPORT template <integral_cast_target_like<int16> To,
    common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi16(+src);
#else
    // Extract low 32 bits of each 64-bit lane
    auto const dwords = _mm_shuffle_epi32(+src, _MM_SHUFFLE(2, 0, 2, 0));
    // Keep only low 16 bits
    auto const masked = _mm_and_si128(dwords, _mm_set1_epi32(0xffff));
    // Pack 32 → 16
    return _mm_packs_epi32(masked, _mm_setzero_si128());
#endif
}

DPL_EXPORT template <integral_cast_target_like<int16> To,
    common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_epi16(+src);
#else
    auto const masked = _mm_and_si128(+src, _mm_set1_epi32(0xffff));
    return _mm_packs_epi32(masked, _mm_setzero_si128());
#endif
}

DPL_EXPORT template <integral_cast_target_like<int16> To,
    common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <integral_cast_target_like<int16> To,
    common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu8_epi16(+src);
    } else {
        return _mm_cvtepi8_epi16(+src);
    }
}

DPL_EXPORT template <integral_cast_target_like<int8> To, floating_point E>
requires (!brain_float<E>) && requires(abi_tag tag, simd<E> src) {
    xmm::cast<signed_representation_t<E>>(tag, src);
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    static_assert(sizeof(E) > sizeof(To));
    return xmm::cast<int8>(xmm::cast<signed_representation_t<E>>(tag, src));
}

DPL_EXPORT template <integral_cast_target_like<int8> To,
    common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <integral_cast_target_like<int8> To,
    common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi16_epi8(+src);
#else
    // Keep only low 8 bits of each 16-bit element
    auto const masked = _mm_and_si128(+src, _mm_set1_epi16(0xff));
    // Pack 16-bit → 8-bit (safe now, no saturation possible)
    return _mm_packs_epi16(masked, _mm_setzero_si128());
#endif
}

DPL_EXPORT template <integral_cast_target_like<int8> To,
    common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_epi8(+src);
#else
    // Keep only low 8 bits of each dword
    auto const masked = _mm_and_si128(+src, _mm_set1_epi32(0xff));
    // 32 → 16
    auto const words = _mm_packs_epi32(masked, _mm_setzero_si128());
    // 16 → 8
    return _mm_packs_epi16(words, _mm_setzero_si128());
#endif
}

DPL_EXPORT template <integral_cast_target_like<int8> To,
    common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi8(+src);
#else
    auto const qwords = _mm_and_si128(+src, _mm_set1_epi64x(0xff));
    // Extract low 32 bits of each 64-bit lane
    auto const dwords = _mm_shuffle_epi32(qwords, _MM_SHUFFLE(3, 1, 2, 0));
    // 32 → 16
    auto const words = _mm_packs_epi32(dwords, _mm_setzero_si128());
    // 16 → 8
    return _mm_packs_epi16(words, _mm_setzero_si128());
#endif
}
} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

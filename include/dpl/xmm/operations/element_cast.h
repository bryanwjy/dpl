// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/arithmetic.h"
#  include "dpl/xmm/operations/bitwise.h"
#  include "dpl/xmm/operations/reinterpret.h"
#  include "dpl/xmm/operations/select.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/infinity.h"
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/numbers/integral_traits.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/std/utility/to_unsigned.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <typename T, typename U>
concept integral_cast_target_like =
    integral<U> && integral<representation_t<T>> && (sizeof(U) == sizeof(T));

namespace details {
template <typename>
struct convert_t;

template <>
struct convert_t<float> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<float> DPL_VECTORCALL operator()(
        vector<float> src) noexcept {
        return src;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<float> DPL_VECTORCALL operator()(
        vector<double> src) noexcept {
        return _mm_cvtpd_ps(+src);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<float> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
#  if DPL_SIMD_X86_F16C
        return _mm_cvtph_ps(+src);
#  else
        auto const arg =
            vector<uint32>(_mm_cvtepu16_epi32(+xmm::reinterpret<uint16>(src)));
        auto const msb = xmm::broadcast<uint32>(0x8000);
        auto const signs = xmm::reinterpret<float>(
            xmm::bwshift_left<16>(xmm::bwand(arg, msb)));
        auto const parg = xmm::bwshift_left<13>(xmm::bwandnot(arg, msb));

        constexpr auto exp_mask = xmm::broadcast<uint32>(0x0f800000);
        auto const isfinite = vector<uint32>(_mm_cmplt_epi32(+parg, +exp_mask));

        constexpr auto inf32 = xmm::broadcast<uint32>(0x7f800000);
        auto const nonfinite = xmm::bwandnot(xmm::bwor(inf32, parg), isfinite);
        constexpr auto shift = xmm::broadcast<float>(0x1p112f);
        auto const shifted =
            xmm::multiply(xmm::reinterpret<float>(parg), shift);

        auto const abs_f32 =
            xmm::bwor(xmm::reinterpret<float>(nonfinite), shifted);
        return xmm::bwor(abs_f32, signs);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<float> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
        return _mm_cvtpbh_ps(+src);
#  else
        auto const isrc = __DPL bit_cast<__m128i>(+src);
        return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(isrc), 16));
#  endif
    }

    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<float>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
            if constexpr (unsigned_integral<representation_t<E>>) {
                return _mm_cvtepu64_ps(+src);
            } else {
                return _mm_cvtepi64_ps(+src);
            }
#  else
            auto const zero = _mm_setzero_si128();
            if constexpr (signed_integral<representation_t<E>>) {
                auto const hisign =
                    _mm_and_si128(_mm_set1_epi64x(1ll << 63), +src);
                auto const losign = _mm_castsi128_ps(
                    _mm_srli_epi64(_mm_castps_si128(hisign), 32));

                src = xmm::abs(src);
                auto const ishizero = _mm_cmpeq_epi32(+src, zero);
                auto const hicorrection = _mm_andnot_si128(ishizero,
                    _mm_or_si128(_mm_set1_epi64x(32ll << 55), hisign));
                auto const islarge =
                    _mm_castsi128_ps(_mm_cmplt_epi32(+src, zero));
                auto const locorrection =
                    _mm_and_ps(islarge, _mm_set1_ps(0x1p32f));
                auto hilohilo = _mm_add_ps(
                    _mm_castsi128_ps(_mm_add_epi64(
                        _mm_castps_si128(_mm_cvtepi32_ps(+src)), hicorrection)),
                    locorrection);
                hilohilo = _mm_xor_ps(hilohilo, losign);
                auto const hihilolo = _mm_castps_si128(_mm_shuffle_ps(
                    hilohilo, hilohilo, _MM_SHUFFLE(3, 1, 2, 0)));
                return _mm_add_ps(
                    _mm_castsi128_ps(_mm_unpacklo_epi64(hihilolo, zero)),
                    _mm_castsi128_ps(_mm_unpackhi_epi64(hihilolo, zero)));
            } else {
                auto const islarge =
                    _mm_castsi128_ps(_mm_cmplt_epi32(+src, zero));
                auto const correction =
                    _mm_and_ps(islarge, _mm_set1_ps(0x1p32f));
                auto const ishizero = _mm_cmpeq_epi32(+src, zero);
                auto const hilohilo = _mm_add_ps(
                    _mm_castsi128_ps(
                        _mm_add_epi64(_mm_castps_si128(_mm_cvtepi32_ps(+src)),
                            _mm_andnot_si128(
                                ishizero, _mm_set1_epi64x(32ll << 55)))),
                    correction);
                auto const hihilolo = _mm_castps_si128(_mm_shuffle_ps(
                    hilohilo, hilohilo, _MM_SHUFFLE(3, 1, 2, 0)));

                return _mm_add_ps(
                    _mm_castsi128_ps(_mm_unpacklo_epi64(hihilolo, zero)),
                    _mm_castsi128_ps(_mm_unpackhi_epi64(hihilolo, zero)));
            }
#  endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            if constexpr (signed_integral<representation_t<E>>) {
                return _mm_cvtepi32_ps(+src);
            } else {
                constexpr auto max =
                    integral_traits<make_signed_t<E>>::max_value;
                auto const gt = _mm_cmpgt_epi32(+src, _mm_set1_epi32(max));
                return _mm_add_ps(_mm_cvtepi32_ps(+src),
                    _mm_and_ps(gt, _mm_set1_ps(0x1p32f)));
            }
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            if constexpr (unsigned_integral<representation_t<E>>) {
                return _mm_cvtepi32_ps(_mm_cvtepu16_epi32(+src));
            } else {
                return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(+src));
            }
        } else {
            if constexpr (unsigned_integral<E>) {
                auto const lo = _mm_cvtepu16_epi32(
                    _mm_unpacklo_epi8(+src, _mm_setzero_si128()));
                return _mm_cvtepi32_ps(lo);
            } else {
                auto const lo = _mm_cvtepi16_epi32(_mm_srai_epi16(
                    _mm_unpacklo_epi8(_mm_setzero_si128(), +src), 8));
                return _mm_cvtepi32_ps(lo);
            }
        }
    }
};

template <integral_cast_target_like<int64> To>
struct convert_t<To> {
    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        if constexpr (sizeof(E) == sizeof(To)) {
            return +src;
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepu32_epi64(+src);
            } else {
                return _mm_cvtepi32_epi64(+src);
            }
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepu16_epi64(+src);
            } else {
                return _mm_cvtepi16_epi64(+src);
            }
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepu8_epi64(+src);
            } else {
                return _mm_cvtepi8_epi64(+src);
            }
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<float> src) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
        if constexpr (unsigned_integral<To>) {
            return _mm_cvttps_epu64(+src);
        } else {
            return _mm_cvttps_epi64(+src);
        }
#  else
        return _mm_unpacklo_epi64(
            _mm_cvtsi64_si128(static_cast<int64>(src[imm<0>])),
            _mm_cvtsi64_si128(static_cast<int64>(src[imm<1>])));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<double> src) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
        if constexpr (unsigned_integral<To>) {
            return _mm_cvttpd_epu64(+src);
        } else {
            return _mm_cvttpd_epi64(+src);
        }
#  else
        return _mm_set_epi64x(
            static_cast<To>(src[imm<1>]), static_cast<To>(src[imm<0>]));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        if constexpr (unsigned_integral<To>) {
            return _mm_cvttph_epu64(+src);
        } else {
            return _mm_cvttph_epi64(+src);
        }
#  else
        constexpr convert_t<float> convert{};
        return operator()(convert(src));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
        constexpr convert_t<float> convert{};
        return operator()(convert(src));
    }
};

template <integral_cast_target_like<int32> To>
struct convert_t<To> {
    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi64_epi32(+src);
#  else
            return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(+src),
                _mm_setzero_ps(), _MM_SHUFFLE(2, 0, 2, 0)));
#  endif
        } else if constexpr (sizeof(E) == sizeof(To)) {
            return +src;
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepu16_epi32(+src);
            } else {
                return _mm_cvtepi16_epi32(+src);
            }
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepu8_epi32(+src);
            } else {
                return _mm_cvtepi8_epi32(+src);
            }
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<float> src) noexcept {
        if constexpr (unsigned_integral<To>) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_cvttps_epu32(+src);
#  else
            constexpr int32 magic = 0x4f000000;
            auto xmm0 = +src;
            auto xmm2 = _mm_castsi128_ps(_mm_set1_epi32(magic));
            auto xmm1 = _mm_cmple_ps(xmm2, xmm0);
            xmm2 = _mm_and_ps(xmm1, xmm2);
            xmm1 = _mm_castps_si128(_mm_slli_epi32(_mm_castps_si128(xmm1), 31));
            xmm0 = _mm_sub_ps(xmm0, xmm2);
            xmm0 = _mm_cvttps_epi32(xmm0);
            return _mm_xor_ps(xmm0, xmm1);
#  endif
        } else {
            return _mm_cvttps_epi32(+src);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<double> src) noexcept {
        if constexpr (unsigned_integral<To>) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_cvttpd_epu32(+src);
#  else
            return _mm_set_epi32(0, 0, static_cast<uint32>(src[imm<1>]),
                static_cast<uint32>(src[imm<0>]));
#  endif
        } else {
            return _mm_cvttpd_epi32(+src);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        if constexpr (signed_integral<To>) {
            return _mm_cvttph_epi32(+src);
        } else {
            return _mm_cvttph_epu32(+src);
        }
#  else
        constexpr convert_t<float> to_fp32{};
        if constexpr (signed_integral<To>) {
            // Cheaper to convert to float first
            return _mm_cvttps_epi32(+to_fp32(src));
        } else {
            constexpr auto mantissa_width = 23u;
            constexpr auto hidden_bit = 1u << mantissa_width;
            constexpr auto bias = 127;
            // There's likely a faster way, but this is just easier :p
            auto const f32 = +to_fp32(src);
            auto const u32 = _mm_castps_si128(f32);
            auto const valid =
                _mm_andnot_si128(_mm_cmplt_epi32(u32, _mm_setzero_si128()),
                    _mm_cmplt_epi32(u32, _mm_set1_epi32(0x7f800000)));
            auto const result = _mm_cvttps_epi32(f32);
            return _mm_castps_si128(_mm_blendv_ps(
                _mm_set1_ps(-0.0f), result, _mm_castsi128_ps(valid)));
        }
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
        constexpr convert_t<float> convert{};
        return operator()(convert(src));
    }
};

template <integral_cast_target_like<int16> To>
struct convert_t<To> {
    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi64_epi16(+src);
#  else
            auto const qwords = _mm_and_si128(+src, _mm_set1_epi64x(0xffff));
            // Extract low 32 bits of each 64-bit lane
            auto const dwords =
                _mm_shuffle_epi32(qwords, _MM_SHUFFLE(3, 1, 2, 0));
            return _mm_packus_epi32(dwords, _mm_setzero_si128());
#  endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi32_epi16(+src);
#  else
            auto const dwords = _mm_and_si128(+src, _mm_set1_epi32(0xffff));
            return _mm_packus_epi32(dwords, _mm_setzero_si128());
#  endif
        } else if constexpr (sizeof(E) == sizeof(To)) {
            return +src;
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepu8_epi16(+src);
            } else {
                return _mm_cvtepi8_epi16(+src);
            }
        }
    }

    template <floating_point E>
    requires (same_as<float, E> || same_as<double, E>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept
    requires requires(convert_t<int32> cvtepi32) { cvtepi32(src); }
    {
        constexpr convert_t<int32> to_int;
        return operator()(to_int(src));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        if constexpr (unsigned_integral<To>) {
            cosntexpr auto cmp_lt_oq = 17; // lessthan, ordered, quiet
            auto const error = xmm::broadcast<ext::float16>(dx::msb);
            auto const neg =
                _mm_cmp_ph(+src, xmm::broadcast<ext::float16>(0), cmp_lt_oq);
            return _mm_mask_blend_ph(neg, _mm_cvttph_epi16(+src), error);
        } else {
            return _mm_cvttph_epi16(+src);
        }
#  else
        constexpr convert_t<int32> to_int32;
        auto const hi =
            vector<ext::float16>(__DPL bit_cast<__m128h>(_mm_unpackhi_epi64(
                __DPL bit_cast<__m128i>(+src), _mm_setzero_si128())));
        auto const lo_mask = _mm_set1_epi32(0xffff);
        auto const left = operator()(to_int32(src));
        auto const right = operator()(to_int32(hi));
        auto result = _mm_unpacklo_epi64(+left, +right);
        if constexpr (unsigned_integral<To>) {
            auto const error = _mm_set1_epi16(static_cast<int16>(0x8000));
            return _mm_blendv_epi8(result, error,
                _mm_cmplt_epi16(
                    __DPL bit_cast<__m128i>(+src), _mm_setzero_si128()));
        } else {
            return result;
        }
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
        constexpr convert_t<int32> to_int32;
        auto const hi =
            vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(_mm_unpackhi_epi64(
                __DPL bit_cast<__m128i>(+src), _mm_setzero_si128())));
        auto const lo_mask = _mm_set1_epi32(0xffff);
        auto const left = operator()(to_int32(src));
        auto const right = operator()(to_int32(hi));
        auto result = _mm_unpacklo_epi64(+left, +right);
        if constexpr (unsigned_integral<To>) {
            auto const error = _mm_set1_epi16(static_cast<int16>(0x8000));
            return _mm_blendv_epi8(result, error,
                _mm_cmplt_epi16(
                    __DPL bit_cast<__m128i>(+src), _mm_setzero_si128()));
        } else {
            return result;
        }
    }
};

template <integral_cast_target_like<int8> To>
struct convert_t<To> {
    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi64_epi8(+src);
#  else
            auto const qwords = _mm_and_si128(+src, _mm_set1_epi64x(0xff));
            // Extract low 32 bits of each 64-bit lane
            auto const dwords =
                _mm_shuffle_epi32(qwords, _MM_SHUFFLE(3, 1, 2, 0));
            auto const zero = _mm_setzero_si128();
            return _mm_packus_epi16(_mm_packus_epi32(dwords, zero), zero);
#  endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi32_epi8(+src);
#  else
            auto const masked = _mm_and_si128(+src, _mm_set1_epi32(0xff));
            auto const zero = _mm_setzero_si128();
            return _mm_packus_epi16(_mm_packus_epi32(masked, zero), zero);
#  endif
        } else if constexpr (sizeof(E) == sizeof(int16)) {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi16_epi8(+src);
#  else
            auto const masked = _mm_and_si128(+src, _mm_set1_epi16(0xff));
            return _mm_packus_epi16(masked, _mm_setzero_si128());
#  endif
        } else {
            static_assert(sizeof(E) == sizeof(To));
            return +src;
        }
    }

    template <floating_point E>
    requires (same_as<float, E> || same_as<double, E>) &&
        requires(convert_t<int32> cvtepi32, vector<E> val) { cvtepi32(val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        constexpr convert_t<int32> cvtepi32;
        return operator()(cvtepi32(src));
    }

    template <floating_point_like E>
    requires (sizeof(E) == sizeof(int16)) &&
        requires(convert_t<int16> cvtepi16, vector<E> val) { cvtepi16(val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<To>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        constexpr convert_t<int16> cvtepi16;
        return operator()(cvtepi16(src));
    }
};

template <>
struct convert_t<double> {

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<double> DPL_VECTORCALL operator()(
        vector<float> src) noexcept {
        return _mm_cvtps_pd(+src);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<double> DPL_VECTORCALL operator()(
        vector<double> src) noexcept {
        return src;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<double> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        return _mm_cvtph_pd(+src);
#  else
        constexpr convert_t<uint64> to_uint;
        auto const arg = to_uint(xmm::reinterpret<uint16>(src));
        auto const msb = xmm::broadcast<uint64>(0x8000);
        auto const signs = xmm::reinterpret<double>(
            xmm::bwshift_left<48>(xmm::bwand(arg, msb)));
        auto const parg = xmm::bwshift_left<42>(xmm::bwandnot(arg, msb));

        constexpr auto exp_mask = xmm::broadcast<uint64>(0x1full << 52);
        auto const isfinite = vector<uint64>(_mm_cmpgt_epi64(+exp_mask, +parg));

        constexpr auto inf64 = xmm::broadcast<uint64>(
            __DPL bit_cast<uint64>(dx::infinity_v<double>));
        auto const nonfinite = xmm::bwandnot(xmm::bwor(inf64, parg), isfinite);
        constexpr auto shift = xmm::broadcast<double>(0x1p1008);
        auto const shifted =
            xmm::multiply(xmm::reinterpret<double>(parg), shift);
        auto const abs_f64 =
            xmm::bwor(xmm::reinterpret<double>(nonfinite), shifted);
        return xmm::bwor(abs_f64, signs);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<double> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
        constexpr convert_t<float> to_float32;
        return operator()(to_float32(src));
    }

    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<double>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
            return _mm_cvtepi64_pd(+src);
#  else
            // Without AVX512, it is simply not worth it to vectorize
            if constexpr (signed_integral<E>) {
                return _mm_setr_pd(
                    static_cast<double>(_mm_extract_epi64(src, 0)),
                    static_cast<double>(_mm_extract_epi64(src, 1)));
            } else {
                return _mm_setr_pd(static_cast<double>( __DPL to_unsigned(
                                       _mm_extract_epi64(src, 0))),
                    static_cast<double>(
                        __DPL to_unsigned(_mm_extract_epi64(src, 1))));
            }
#  endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            if constexpr (signed_integral<E>) {
                return _mm_cvtepi32_pd(+src);
            } else {
                auto const large = _mm_cmplt_epi32(+src, _mm_setzero_si128());
                auto const islolarge =
                    _mm_castsi128_pd(_mm_unpacklo_epi32(large, large));
                return _mm_add_pd(_mm_and_pd(islolarge, _mm_set1_pd(0x1p32)),
                    _mm_cvtepi32_pd(+src));
            }
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            if constexpr (unsigned_integral<E>) {
                return _mm_cvtepi32_pd(_mm_cvtepu16_epi32(+src));
            } else {
                return _mm_cvtepi32_pd(_mm_cvtepi16_epi32(+src));
            }
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            if constexpr (unsigned_integral<E>) {
                auto const lo = _mm_cvtepu16_epi32(
                    _mm_unpacklo_epi8(+src, _mm_setzero_si128()));
                return _mm_cvtepi32_pd(lo);
            } else {
                auto const lo = _mm_cvtepi16_epi32(_mm_srai_epi16(
                    _mm_unpacklo_epi8(_mm_setzero_si128(), +src), 8));
                return _mm_cvtepi32_pd(lo);
            }
        }
    }
};

template <>
struct convert_t<ext::float16> {
private:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::float16> DPL_VECTORCALL to_postive_inthalf(
        vector<float> f32) noexcept {
        auto const bits = xmm::reinterpret<uint32>(f32);
        auto const mantissa = xmm::bwand(
            xmm::bwshift_right<13>(bits), xmm::broadcast<uint32>(0x3ff));
        auto const exp =
            _mm_subs_epu16(+xmm::bwshift_right<23>(bits), _mm_set1_epi32(112));
        auto const bexp = vector<uint32>(_mm_slli_epi32(exp, 10));
        return xmm::reinterpret<ext::float16>(
            xmm::select(mask<float>(_mm_cmpge_ps(+f32, _mm_set1_ps(0x1p16f))),
                xmm::broadcast<uint32>(0x7c00), xmm::bwor(bexp, mantissa)));
    }

public:
    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::float16>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        constexpr convert_t<float> to_fp32;
        if constexpr (sizeof(E) == sizeof(int64)) {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
            if constexpr (unsigned_integral<representation_t<E>>) {
                return _mm_cvtepu64_ph(+src);
            } else {
                return _mm_cvtepi64_ph(+src);
            }
#  else
            auto const zero = _mm_setzero_si128();
            if constexpr (unsigned_integral<representation_t<E>>) {
                auto const all = _mm_set1_epi64x(0xffff);
                auto const isinf = mask<uint16>(_mm_packs_epi32(
                    _mm_packs_epi32(_mm_cmpgt_epi64(+src, all), zero), zero));
                auto const trunc32 = vector<int32>(
                    _mm_packus_epi32(_mm_and_si128(+src, all), zero));
                auto const trunc = to_postive_inthalf(to_fp32(trunc32));
                return xmm::select(
                    isinf, xmm::broadcast<ext::float16>(dx::infinity), trunc);
            } else {
                auto const abs = xmm::reinterpret<uint64>(xmm::abs(src));
                auto const vsign16 = vector<uint16>(_mm_and_si128(
                    _mm_packs_epi32(_mm_packs_epi32(+src, zero), zero),
                    _mm_set1_epi16(0x8000)));

                return xmm::reinterpret<ext::float16>(xmm::bit_fill(
                    vsign16, operator()(xmm::reinterpret<uint64>(abs))));
            }
#  endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
            if constexpr (unsigned_integral<representation_t<E>>) {
                return _mm_cvtepu32_ph(+src);
            } else {
                return _mm_cvtepi32_ph(+src);
            }
#  else
            auto const zero = _mm_setzero_si128();
            if constexpr (unsigned_integral<representation_t<E>>) {
                auto const all = _mm_set1_epi32(0xffff);
                auto const isinf = mask<uint16>(
                    _mm_packs_epi32(_mm_cmpgt_epi32(+src, all), zero));
                auto const clamped = vector<E>(_mm_and_si128(+src, all));
                auto const trunc = to_postive_inthalf(to_fp32(clamped));
                return xmm::select(xmm::abi, isinf,
                    xmm::broadcast<ext::float16>(xmm::abi, dx::infinity),
                    trunc);
            } else {
                auto const abs =
                    xmm::reinterpret<uint32>(xmm::abs(xmm::abi, src));
                auto const vsign16 = vector<uint16>(_mm_and_si128(
                    _mm_packs_epi32(+src, zero), _mm_set1_epi16(0x8000)));
                return xmm::reinterpret<ext::float16>(xmm::abi,
                    xmm::bit_fill(xmm::abi,
                        vsign16,
                        operator()(xmm::reinterpret<uint32>(xmm::abi, abs))));
            }
#  endif
        } else if constexpr (sizeof(E) == sizeof(int16)) {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
            if constexpr (unsigned_integral<representation_t<E>>) {
                return _mm_cvtepu16_ph(+src);
            } else {
                return _mm_cvtepi16_ph(+src);
            }
#  else
            if constexpr (unsigned_integral<representation_t<E>>) {
                auto const zero = _mm_setzero_si128();
                auto const lo =
                    to_fp32(vector<uint32>(_mm_unpacklo_epi16(+src, zero)));
                auto const hi =
                    to_fp32(vector<uint32>(_mm_unpackhi_epi16(+src, zero)));
                auto const lo16 =
                    xmm::reinterpret<float>(xmm::abi, to_postive_inthalf(lo));
                auto const hi16 =
                    xmm::reinterpret<float>(xmm::abi, to_postive_inthalf(hi));
                using simdf = remove_const_t<decltype(lo)>;
                return xmm::reinterpret<ext::float16>(xmm::abi,
                    simdf(_mm_shuffle_ps(lo16, hi16, _MM_SHUFFLE(1, 0, 1, 0))));
            } else {
                auto const sign = xmm::reinterpret<ext::float16>(xmm::abi,
                    vector<uint32>(
                        _mm_and_si128(+src, _mm_set1_epi16(0x8000))));
                auto const abs = operator()(xmm::reinterpret<uint16>(
                    xmm::abi, xmm::abs(xmm::abi, src)));
                return xmm::bwor(xmm::abi, sign, abs);
            }
#  endif
        } else {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
            constexpr convert_t<int16> to_i16;
            return _mm_cvtepi16_ph(+to_i16(src));
#  else
            if constexpr (unsigned_integral<E>) {
                constexpr convert_t<uint16> to_uint16;
                return operator()(to_uint16(src));
            } else {
                constexpr convert_t<int16> to_int16;
                return operator()(to_int16(src));
            }
#  endif
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::float16> DPL_VECTORCALL operator()(
        vector<double> src) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        return _mm_cvtpd_ph(+src);
#  else
        auto const i64 = xmm::reinterpret<int64>(src);
        auto const msb64 = xmm::broadcast<int64>(dx::msb);
        auto const sign16 = xmm::bwshift_right<48>(
            xmm::reinterpret<uint64>(xmm::bwand(msb64, i64)));
        auto const abs = xmm::reinterpret<double>(xmm::bwandnot(i64, msb64));
        auto const shifted = xmm::reinterpret<uint64>(
            xmm::multiply(abs, xmm::broadcast<double>(0x1p-1008)));
        constexpr auto round_mask = (0x1ull << 41) - 1ull;
        auto const rounded = xmm::bwshift_right<42>(
            xmm::add(xmm::bwand(xmm::bwshift_right<42>(shifted),
                         xmm::broadcast<uint64>(dx::one)),
                xmm::add(shifted, xmm::broadcast<uint64>(round_mask))));
        auto const limit = xmm::broadcast<double>(0x1p16);
        auto const isinf = mask<double>(_mm_cmpge_pd(+abs, +limit));
        auto const isnan =
            xmm::bwand(vector<double>(_mm_cmpunord_pd(+src, +src)),
                xmm::reinterpret<double>(xmm::broadcast<int64>(0x7fff)));

        constexpr auto infval = static_cast<int64>(
            __DPL bit_cast<int16>(dx::infinity_v<ext::float16>));
        auto const inf16 =
            xmm::reinterpret<ext::float16>(xmm::broadcast<int64>(infval));
        auto const f16 = xmm::select(isinf, xmm::reinterpret<double>(inf16),
            xmm::reinterpret<double>(rounded));
        auto const result =
            xmm::bwor(sign16, xmm::reinterpret<uint64>(xmm::bwor(isnan, f16)));
        auto const zero = _mm_setzero_si128();
        // The cast here is just to ditribute the bits into place
        return xmm::reinterpret<ext::float16>(xmm::abi,
            vector<ext::float16>(_mm_packus_epi32(
                _mm_shuffle_ps(_mm_castsi128_ps(+result),
                    _mm_castsi128_ps(zero), _MM_SHUFFLE(2, 0, 2, 0)),
                zero)));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::float16> DPL_VECTORCALL operator()(
        vector<float> src) noexcept {
#  if DPL_SIMD_X86_F16C
        return __DPL bit_cast<__m128h>(
            _mm_cvtps_ph(+src, _MM_FROUND_TO_NEAREST_INT));
#  else
        auto const i32 = xmm::reinterpret<int32>(src);
        auto const msb32 = xmm::broadcast<int32>(dx::msb);
        auto const sign16 = xmm::bwshift_right<16>(
            xmm::reinterpret<uint32>(xmm::bwand(msb32, i32)));
        auto const abs = xmm::reinterpret<float>(xmm::bwandnot(i32, msb32));
        auto const shifted = xmm::reinterpret<uint32>(
            xmm::multiply(abs, xmm::broadcast<float>(0x1p-112f)));

        auto const rounded = xmm::bwshift_right<13>(
            xmm::add(xmm::bwand(xmm::bwshift_right<13>(shifted),
                         xmm::broadcast<uint32>(dx::one)),
                xmm::add(shifted, xmm::broadcast<uint32>(0xfffu))));

        auto const limit = xmm::broadcast<float>(0x1p16f);
        auto const isinf = mask<float>(_mm_cmpge_ps(+abs, +limit));
        auto const isnan =
            xmm::bwandnot(vector<float>(_mm_cmpunord_ps(+src, +src)),
                xmm::reinterpret<float>(msb32));

        constexpr auto infval = static_cast<int32>(
            __DPL bit_cast<int16>(dx::infinity_v<ext::float16>));
        auto const inf16 =
            xmm::reinterpret<ext::float16>(xmm::broadcast<int32>(infval));
        auto const f16 = xmm::select(isinf, xmm::reinterpret<float>(inf16),
            xmm::reinterpret<float>(rounded));
        auto const result =
            xmm::bwor(sign16, xmm::reinterpret<uint32>(xmm::bwor(isnan, f16)));
        return xmm::reinterpret<ext::float16>(
            vector<uint32>(_mm_packus_epi32(+result, _mm_setzero_si128())));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::float16> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
        return src;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::float16> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
        constexpr convert_t<float> to_f32;
        auto const hi =
            vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(_mm_unpackhi_epi64(
                __DPL bit_cast<__m128i>(+src), _mm_setzero_si128())));
        auto const left = operator()(to_f32(src));
        auto const right = operator()(to_f32(hi));
        return __DPL bit_cast<__m128h>(
            _mm_unpacklo_epi64(__DPL bit_cast<__m128i>(+left),
                __DPL bit_cast<__m128i>(+right)));
    }
};

template <>
struct convert_t<ext::bfloat16> {
    template <integral E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::bfloat16>
        DPL_VECTORCALL operator()(vector<E> src) noexcept {
        constexpr convert_t<float> to_fp32;
        if constexpr (sizeof(E) == sizeof(int64)) {
            return operator()(to_fp32(src));
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            return operator()(to_fp32(src));
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            auto const hi =
                vector<E>(_mm_unpackhi_epi64(+src, _mm_setzero_si128()));
            auto const left = operator()(to_fp32(src));
            auto const right = operator()(to_fp32(hi));
            return __DPL bit_cast<__m128bh>(
                _mm_unpacklo_epi64(__DPL bit_cast<__m128i>(+left),
                    __DPL bit_cast<__m128i>(+right)));
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            constexpr convert_t<int16> to_int16;
            return operator()(to_int16(src));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::bfloat16> DPL_VECTORCALL operator()(
        vector<double> src) noexcept {
        constexpr convert_t<float> to_fp32;
        return operator()(to_fp32(src));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::bfloat16> DPL_VECTORCALL operator()(
        vector<float> src) noexcept {
#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
        return _mm_cvtneps_pbh(+src);
#  else
        // deal with nan & inf
        auto const fval = +src;
        auto const ival = __DPL bit_cast<__m128i>(fval);
        auto const zero = _mm_setzero_si128();
        auto const isnan = _mm_unpacklo_epi16(
            _mm_castps_si128(_mm_cmpunord_ps(fval, fval)), zero);
        // round to nearest even
        auto const lsb =
            _mm_and_si128(_mm_srli_epi32(ival, 16), _mm_set1_epi32(1));
        auto const low = _mm_set1_epi32(0x7fff);
        auto const bias = _mm_add_epi32(low, lsb);
        auto const result =
            _mm_or_si128(_mm_srli_epi32(_mm_add_epi32(ival, bias), 16), isnan);
        return __DPL bit_cast<__m128bh>(_mm_packus_epi32(result, zero));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::bfloat16> DPL_VECTORCALL operator()(
        vector<ext::bfloat16> src) noexcept {
        return src;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static inline vector<ext::bfloat16> DPL_VECTORCALL operator()(
        vector<ext::float16> src) noexcept {
        constexpr convert_t<float> to_f32;
        auto const hi =
            vector<ext::float16>(__DPL bit_cast<__m128h>(_mm_unpackhi_epi64(
                __DPL bit_cast<__m128i>(+src), _mm_setzero_si128())));
        auto const left = operator()(to_f32(src));
        auto const right = operator()(to_f32(hi));
        return __DPL bit_cast<__m128bh>(
            _mm_unpacklo_epi64(__DPL bit_cast<__m128i>(+left),
                __DPL bit_cast<__m128i>(+right)));
    }
};

template <typename>
struct convert_t {};

} // namespace details

template <simd_element To, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<E> src) noexcept
requires requires(xmm::details::convert_t<To> cvt) { cvt(src); }
{
    constexpr xmm::details::convert_t<To> cvt;
    return cvt(src);
}

template <simd_element_for<xmm::abi_tag> To, simd_element_for<xmm::abi_tag> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(abi_tag, vector<E> src) noexcept
requires requires(xmm::details::convert_t<To> cvt) { cvt(src); }
{
    constexpr xmm::details::convert_t<To> cvt;
    return cvt(src);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<float> to_float(vector<ext::bfloat16> arg) noexcept {
    return xmm::element_cast<float>(arg);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif

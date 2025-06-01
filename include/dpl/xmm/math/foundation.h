// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/fwd.h"

#include "dpl/xmm/concepts.h"
#include "dpl/xmm/constants.h"
#include "dpl/xmm/native.h"
#include "dpl/xmm/scalar.h"
#include "dpl/xmm/select.h"
#include "dpl/xmm/vector.h"
#include "dpl/xmm/vector_mask.h"

#if !DPL_MODULES
#  include <immintrin.h>

#  include <climits>
#  include <concepts>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT_BEGIN
template <xmm_element T>
requires std::signed_integral<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> abs(
    vector<T> val) noexcept {
    if constexpr (sized_signed_integral<T, 1>) {
        return _mm_abs_epi8(+val);
    } else if constexpr (sized_signed_integral<T, 2>) {
        return _mm_abs_epi16(+val);
    } else if constexpr (sized_signed_integral<T, 4>) {
        return _mm_abs_epi32(+val);
    } else {
        static_assert(sized_signed_integral<T, 8>, "unsupported operation");
        return select(val, vector{zero<T>} - val, vector_mask{val});
    }
}

template <xmm_element T>
requires std::floating_point<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> abs(
    vector<T> val) noexcept {
    return ~vector{msb<T>} & val;
}

template <xmm_scalar_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline scalar<T> abs(
    scalar<T> val) noexcept {
    static_assert(std::floating_point<T>, "unsupported operation");

    static constexpr auto to_clear = sizeof(T) * CHAR_BIT - 1;
    static constexpr auto mask = (~bitset_for<b128>{}).clear(to_clear);
    return reinterpret_as<T>(vector{+mask} & reinterpret_as<b128>(vector{val}));
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> rsqrt(
    vector<T> val) noexcept {
#if DPL_SIMD_X86_AVX512 & DPL_SIMD_X86_AVX512VL
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rsqrt14_ps(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rsqrt14_pd(+val);
    }
#else
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rsqrt_ps(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rsqrt_pd(+val);
    }
#endif
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> rsqrt(
    scalar<T> val) noexcept {
#if DPL_SIMD_X86_AVX512
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rsqrt14_ss(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rsqrt14_sd(+val);
    }
#else
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rsqrt_ss(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rsqrt_sd(+val);
    }
#endif
}

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
template <sized_floating_point<2> T>
requires std::same_as<float16, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> rsqrt(
    vector<T> val) noexcept {
    return _mm_rsqrt_ph(+val);
}

template <sized_floating_point<2> T>
requires std::same_as<float16, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> rsqrt(
    scalar<T> val) noexcept {
    return _mm_rsqrt_sh(+val);
}
#endif

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> rcp(
    vector<T> val) noexcept {
#if DPL_SIMD_X86_AVX512 & DPL_SIMD_X86_AVX512VL
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rcp14_ps(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rcp14_pd(+val);
    }
#else
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rcp_ps(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rcp_pd(+val);
    }
#endif
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> rcp(
    scalar<T> val) noexcept {
#if DPL_SIMD_X86_AVX512
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rcp14_ss(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rcp14_sd(+val);
    }
#else
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_rcp_ss(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_rcp_sd(+val);
    }
#endif
}

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
template <sized_floating_point<2> T>
requires std::same_as<float16, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> rcp(
    vector<T> val) noexcept {
    return _mm_rcp_ph(+val);
}

template <sized_floating_point<2> T>
requires std::same_as<float16, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> rcp(
    scalar<T> val) noexcept {
    return _mm_rcp_sh(+val);
}
#endif

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> sqrt(
    vector<T> val) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_sqrt_ps(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_sqrt_pd(+val);
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> sqrt(
    scalar<T> val) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_sqrt_ss(+val);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_sqrt_sd(+val);
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline auto operator&(
    vector<T> self, exponent_t) noexcept
requires std::floating_point<T>
{
    if constexpr (sized_floating_point<T, 4>) {
        auto const mask{0x7f8};
        auto const shift = reinterpret_as<int>(self) >> immediate_v<23>;
        return shift & mask;
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        auto const mask{0x7ffll};
        auto const shift = reinterpret_as<long long>(self) >> immediate_v<52>;
        return shift & mask;
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline auto operator&(
    vector<T> self, mantissa_t) noexcept
requires std::floating_point<T>
{
    if constexpr (sized_floating_point<T, 4>) {
        static constexpr auto mask = ((1 << 23) - 1);
        return reinterpret_as<int>(self) & vector{mask};
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        static constexpr auto mask = ((1ll << 52) - 1);
        return reinterpret_as<long long>(self) & vector{mask};
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline auto operator&(
    vector<T> self, signbit_t) noexcept
requires std::floating_point<T>
{
    if constexpr (sized_floating_point<T, 4>) {
        return reinterpret_as<unsigned int>(self) >> immediate_v<31>;
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return reinterpret_as<unsigned long long>(self) >> immediate_v<63>;
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector_mask<T> isinf(
    vector<T> val) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        static constexpr auto exponent_bits = 0x7f800000;
        return reinterpret_as<int>(abs(val)) == vector{exponent_bits};
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        static constexpr auto exponent_bits = 0x7ff0000000000000ll;
        return reinterpret_as<long long>(abs(val)) == vector{exponent_bits};
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar_mask<T> isinf(
    scalar<T> val) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        static constexpr auto exponent_bits = 0x7f800000;
        return reinterpret_as<int>(abs(val)) == scalar{exponent_bits};
    } else {
        static constexpr auto exponent_bits = 0x7ff0000000000000ll;
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return reinterpret_as<long long>(abs(val)) == scalar{exponent_bits};
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector_mask<T> isnan(
    vector<T> val) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return vector_mask<T>{_mm_cmpunord_ps(+val)};
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return vector_mask<T>{_mm_cmpunord_pd(+val)};
    }
}

template <std::floating_point T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar_mask<T> isnan(
    scalar<T> val) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return scalar_mask<T>{_mm_cmpunord_ss(+val)};
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return scalar_mask<T>{_mm_cmpunord_sd(+val)};
    }
}

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END
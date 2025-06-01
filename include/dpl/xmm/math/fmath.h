// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/fwd.h"

#include "dpl/numbers.h"
#include "dpl/xmm/concepts.h"
#include "dpl/xmm/constants.h"
#include "dpl/xmm/convert.h"    // IWYU pragma: export
#include "dpl/xmm/foundation.h" // IWYU pragma: export
#include "dpl/xmm/native.h"
#include "dpl/xmm/round.h" // IWYU pragma: export
#include "dpl/xmm/scalar.h"
#include "dpl/xmm/select.h"
#include "dpl/xmm/vector.h"
#include "dpl/xmm/vector_mask.h"

#if !DPL_MODULES
#  include <immintrin.h>

#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

namespace details::fmath {
template <typename T>
concept native_float_type =
    std::same_as<scalar<float>, T> || std::same_as<vector<float>, T>;
template <typename T>
concept native_int_type =
    std::same_as<scalar<int>, T> || std::same_as<vector<int>, T>;

template <native_int_type I>
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) rebind_simd_t<I, float> pow2(
    I exp) {
    return reinterpret_as<float>(exp + I{0x7f} << immediate_v<23>);
}

template <native_float_type F>
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) auto ldexp_unsafe(
    F mantissa, rebind_simd_t<F, int> exp) noexcept {
    return reinterpret_as<float>(
        reinterpret_as<int>(mantissa) + (exp << immediate_v<23>));
}

template <native_float_type F>
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) auto ldexp_fast(
    F digits, rebind_simd_t<F, int> exp) noexcept {
    auto const half_exp = vector{exp} >> immediate_v<1>;
    return digits * pow2(half_exp) * pow2(exp - half_exp);
}

template <native_float_type F>
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) rebind_simd_t<F, int> ilogbkf(
    details::native_float_type F d) noexcept {
    static constexpr float low = 5.421010862427522E-20f;
    static constexpr float multiplier = 1.8446744073709552E19f;
    auto const mask = d < vector{low};
    d = select(vector{1.0F}, vector{multiplier}, mask) * d;
    auto q =
        reinterpret_as<int>(reinterpret_as<unsigned>(d) >> immediate_v<23>) &
        vector{0xff};
    auto const offset = select(vector{0}, vector{64}, mask);
    q = q - vector{0x7f};
    q = q - offset;
    return q;
}

template <native_float_type F>
DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) rebind_simd_t<F, int> ilogb2kf(
    details::native_float_type F d) noexcept {
    auto ints =
        reinterpret_as<int>(reinterpret_as<unsigned>(d) >> immediate_v<23>);
    ints &= vector{all<int>};
    ints -= vector{0x7f};
    return ints;
}
} // namespace details::fmath

DPL_EXPORT_BEGIN

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F ilogb(F d) noexcept;
template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F cos(F d) noexcept;

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F sin(F d) noexcept;

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F tan(F d) noexcept;
template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F acos(F d) noexcept;

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F asin(F d) noexcept;

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F atan(F d) noexcept;
template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F atan2(F a, F b) noexcept;

extern template vector<float> ilogb(vector<float>) noexcept;
extern template scalar<float> ilogb(scalar<float>) noexcept;
extern template vector<float> cos(vector<float>) noexcept;
extern template scalar<float> cos(scalar<float>) noexcept;
extern template vector<float> sin(vector<float>) noexcept;
extern template scalar<float> sin(scalar<float>) noexcept;
extern template vector<float> tan(vector<float>) noexcept;
extern template scalar<float> tan(scalar<float>) noexcept;
extern template vector<float> acos(vector<float>) noexcept;
extern template scalar<float> acos(scalar<float>) noexcept;
extern template vector<float> asin(vector<float>) noexcept;
extern template scalar<float> asin(scalar<float>) noexcept;
extern template vector<float> atan(vector<float>) noexcept;
extern template scalar<float> atan(scalar<float>) noexcept;
extern template vector<float> atan2(vector<float>, vector<float>) noexcept;
extern template scalar<float> atan2(scalar<float>, scalar<float>) noexcept;

DPL_EXPORT_END

namespace details::fmath {
struct cos {
    static constexpr alignas(16) float large_poly[4] = {3.140625f * -0.5f,
        0.0009670257568359375f * -0.5f, 6.2771141529083251953e-07f * -0.5f,
        1.2154201256553420762e-10f * -0.5f};
    static constexpr alignas(16) float small_poly[4] = {
        3.1414794921875f * -0.5f, 0.00011315941810607910156f * -0.5f,
        1.9841872589410058936e-09f * -0.5f, 0.0f};

    static constexpr alignas(16) float final_poly[4] = {
        2.6083159809786593541503e-06f, -0.0001981069071916863322258f,
        0.00833307858556509017944336f, -0.166666597127914428710938f};
};

template <typename T>
struct rempi_result;

template <>
struct rempi_result<vector<float>> {
    vector<float> df;
    vector<int> i;
};

template <>
struct rempi_result<scalar<float>> {
    scalar<float> df;
    scalar<int> i;
};

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) details::fmath::rempi_result<F> rempi(
    F d) noexcept {
    using float_type = F;
    using int_type = rebind_simd_t<F, int>;
    float_type x, y;
    auto exponent = details::fmath::ilogb2kf(d) - int_type{25};
    auto const q = int_type{exponent > int_type{65}} & int_type{-64};
    d = details::fmath::ldexp_unsafe(a, q);
    exponent &= ~(exponent >> immediate_v<31>);
    exponent <<= immediate_v<2>;
    // TODO
    assert(false);
}

} // namespace details::fmath

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F ilogb(F d) noexcept {
    using float_type = F;
    using int_type = rebind_simd_t<F, int>;
    auto output = details::fmath::ilogbkf(abs(d));
    output = select(output, vector{msb<int>}, d == vector{0.0f});
    output =
        select(output, vector{numeric::maximum_v<int>}, isnan(d) | isinf(d));
    return output;
}

template <details::fmath::native_float_type F>
DPL_ATTRIBUTES(VECTORCALL, CONST, NODISCARD) F cos(F d) noexcept {
    using float_type = F;
    using int_type = rebind_simd_t<F, int>;
    float_type const r = d;
    float_type const n_one = vector{-1.0F};
    int_type const q = int_type{lsb<int>} +
        (convert<int>(round<rounding::zero | rounding::noexc>(
             d * vector{numbers::inv_pi_v<float>} +
             ((d - vector{numbers::half_pi_v<float>} < n_one) & n_one)))
            << immediate_v<1>);

    float_type const qf = convert<float>(q);
    float_type const low = [&]() {
        float_type poly{aligned, details::fmath::cos::small_poly};
        return qf * broadcast<0>(poly) + qf * broadcast<1>(poly) +
            qf * broadcast<2>(poly) + qf * broadcast<3>(poly) +
            float_type{4.0f} * d;
    }();

    auto mask = abs(r) < float_type{125.0f};
    if (!all_of(mask)) {
        float_type const if_high = [&]() {
            float_type const poly{aligned, details::fmath::cos::large_poly};
            return qf * broadcast<0>(poly) + qf * broadcast<1>(poly) +
                qf * broadcast<2>(poly) + qf * broadcast<3>(poly) +
                float_type{4.0f} * d;
        }();

        d = select(low, if_high, mask);
        mask = abs(r) < float_type{39000.0f};
        if (!all_of(mask)) {
            auto const dfi = details::fmath::rempi(d);
            int_type q2 = reinterpret_as<int>(dfi.df) & vector{3};
            q2 =
                q2 + q2 + select(vector{1}, vector{2}, dfi.df.x > vector{0.0F});
            q2 >>= immediate_v<2>;
            auto const o = (dfi.i & vector{1}) == vector{0};
            assert(false);
        }
    } else {
        d = low;
    }

    float_type s = d * d;
    static constexpr auto negation_mask = [](auto q) {
        return vector{(q & int_type{2}) == int_type{zero<int>}}
        << immediate<31>;
    };
    d = negation_mask(q) ^ d; // if q & 2 == 0: d = -d

    float_type const poly = [&]() {
        float_type const s2 = s * s;
        float_type const s3 = s2 * s;
        float_type const s4 = s2 * s2;
        float_type const poly{aligned, details::fmath::cos::final_poly};

        return s * broadcast<3>(poly) + s2 * broadcast<2>(poly) +
            s3 * broadcast<1>(poly) + s4 * broadcast<0>(poly);
    }();

    return poly * d + d;
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END
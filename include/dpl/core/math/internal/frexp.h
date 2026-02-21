// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/ldexp.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/to_integral.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

enum class exp_type {
    intergral,
    floating_point,
};

enum class fr_sign {
    copy,
    positive,
    nan_ifltz
};

enum class fr_interval {
    canonical, // [1,2)
    cmath,     // [0.5,1)
    wide,      // [0.5,2)
    reduced    // [0.75,1.5)
};

template <floating_point E, simd_abi A, exp_type T = exp_type::intergral>
struct frexp_result {
    basic_simd<E, A> fr;
    basic_simd<E, A> exp;
};

template <floating_point E, simd_abi A>
struct frexp_result<E, A, exp_type::intergral> {
    basic_simd<E, A> fr;
    basic_simd<dx::to_signed_integral_t<E>, A> exp;
};

template <floating_point E>
inline constexpr auto denormalizer = []() {
    if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
        return static_cast<E>(0x1.p64);
    } else {
        return static_cast<E>(0x1.p12);
    }
}();
template <floating_point E>
inline constexpr auto subnormal_offset = []() {
    if constexpr (dx::digits_v<E> >= dx::digits_v<float>) {
        return 64;
    } else {
        return 12;
    }
}();

namespace fr {
inline constexpr auto copysign = imm<fr_sign::copy>;
inline constexpr auto positive = imm<fr_sign::positive>;
inline constexpr auto nan_ifltz = imm<fr_sign::nan_ifltz>;
inline constexpr auto canonical = imm<fr_interval::canonical>;
inline constexpr auto cmath = imm<fr_interval::cmath>;
inline constexpr auto wide = imm<fr_interval::wide>;
inline constexpr auto reduced = imm<fr_interval::reduced>;
inline constexpr auto fpexp = imm<exp_type::floating_point>;
} // namespace fr

template <floating_point E, simd_abi A, fr_interval N = fr_interval::cmath,
    exp_type TE = exp_type::intergral, fr_sign S = fr_sign::copy>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr frexp_result<E, A, TE> DPL_VECTORCALL
    frexp(basic_simd<E, A> val, immediate<N> interval = {},
        immediate<TE> etype = {}, immediate<S> sign = {}) noexcept {
    if not consteval {
        if constexpr (requires {
                          frexp(internal::abi<A>, val, etype, sign, interval);
                      }) {
            auto const [fr, exp] =
                frexp(internal::abi<A>, val, etype, sign, interval);
            return frexp_result<E, A, TE>{
                .fr = fr,
                .exp = exp,
            };
        }
    }

    auto const issubnormal = [](auto val) {
        if constexpr (S == fr_sign::nan_ifltz) {
            return val < dx::min_value;
        } else {
            return (val & dx::exponent_bits) == dx::zero;
        }
    }(val);
    auto const dval = [](auto issubnormal, auto val) {
        auto const dval = dx::select(issubnormal, val * denormalizer<E>, val);
        if constexpr (S == fr_sign::positive) {
            return dx::abs(val);
        } else {
            return val;
        }
    }(issubnormal, val);

    if constexpr (N == fr_interval::reduced) {
        constexpr auto fourthirds = dx::broadcast<E, A>(1.0 / 0.75);
        auto const exp = [](auto issubnormal, auto exp) {
            return dx::select(issubnormal, exp - subnormal_offset<E>, exp);
        }(issubnormal, fmath::ilogb(compliance::unsafe, dval * fourthirds));
        auto const fr = [](auto val, auto fr) {
            if constexpr (S == fr_sign::nan_ifltz) {
                return dx::bit_fill(val < dx::zero, fr);
            } else {
                return fr;
            }
        }(fmath::ldexp(compliance::unsafe, dval, -exp));
        if constexpr (TE == exp_type::intergral) {
            return frexp_result<E, A, TE>{
                .fr = fr,
                .exp = exp,
            };
        } else {
            return frexp_result<E, A, TE>{
                .fr = fr,
                .exp = dx::cast<E>(exp),
            };
        }
    } else if constexpr (N == fr_interval::cmath) {
        using int_type = dx::to_signed_integral_t<E>;
        constexpr auto exp_bits =
            __DPL bit_cast<int_type>(dx::exponent_bits_v<E>);
        constexpr auto magic =
            dx::broadcast<int_type, A>(exponent_bias_v<E> - 1);
        constexpr auto magic_exp =
            __DPL bit_cast<E>(magic << dx::mantissa_width_v<E>);

        auto const exp = [&]() {
            auto const mexp =
                dx::reinterpret<int_type>(val & dx::exponent_bits);
            auto const exp_offset =
                dx::bit_drop(mexp == dx::zero || mexp == exp_bits, magic);
            auto const exp = (mexp >> imm<dx::mantissa_width_v<E>>)-exp_offset;

            return dx::select(issubnormal, exp - subnormal_offset<E>, exp);
        }();
        auto const fr = [&]() {
            auto const fr = (val & ~exponent_bits) | magic_exp;
            if constexpr (S == fr_sign::nan_ifltz) {
                return dx::bit_fill(val < dx::zero, fr);
            } else {
                return fr;
            }
        }();
        if constexpr (TE == exp_type::intergral) {
            return frexp_result<E, A, TE>{
                .fr = fr,
                .exp = exp,
            };
        } else {
            return frexp_result<E, A, TE>{
                .fr = fr,
                .exp = dx::cast<E>(exp),
            };
        }
    } else {
        static_assert(sizeof(E) == 0, "Unimplemented");
    }
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END

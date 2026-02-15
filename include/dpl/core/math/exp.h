// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h" // IWYU pragma: keep
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void exp(...) noexcept = delete;

template <typename T>
concept unqualified_exp = floating_point_simd<T> && requires(T val) {
    { exp(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

struct exp_t {
private:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        using sint = to_signed_integral_t<float>;
        auto const q = dx::cast<sint>(val * fmath::inv_ln2);
        auto const qf = dx::cast<float>(q);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<1.0f, 1.0f,
            0.5f,                          //
            0.166666671633720397949219f,   //
            0.0416664853692054748535156f,  //
            0.00833336077630519866943359f, //
            0.00139304355252534151077271f, //
            0.000198527617612853646278381f>
            polynomial;
        // x2 * f + x + 1
        auto u = polynomial(s);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val > 100.0f, dx::infinity, u);
        // underflow
        return dx::bit_keep(val >= -104.0f, u);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<double, A> val) noexcept {
        using sint = to_signed_integral_t<double>;
        auto const q = dx::cast<sint>(val * fmath::inv_ln2);
        auto const qf = dx::cast<double>(q);
        using fpair = fmath::pair<double, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<1.0, 1.0, 0.5,
            0.1666666666666669072e+0, 0.4166666666666602598e-1,
            0.8333333333314938210e-2, 0.1388888888914497797e-2,
            0.1984126989855865850e-3, 0.2480158687479686264e-4,
            0.2755723402025388239e-5, 0.2755762628169491192e-6,
            0.2511210703042288022e-7, 0.2081276378237164457e-8>
            polynomial;
        auto u = polynomial(s);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        static constexpr auto max_log = 0x1.62e42fefa39efp+9;
        u = dx::select(val > max_log, dx::infinity, u);
        // underflow
        return dx::bit_keep(val >= -1000.0, u);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (unqualified_exp<T>) {
            if not consteval {
                return exp(internal::abi<T>, val);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept
        -> equivalent_simd_as<T> auto {
        if constexpr (unqualified_exp<T>) {
            return exp(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::exp_t exp{};
}
DPL_DEFAULT_NAMESPACE_END

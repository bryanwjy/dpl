// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/round.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
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
    template <floating_point E, simd_abi A>
    requires (dx::digits_v<E> < dx::digits_v<float>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        using sint = signed_representation_t<float>;
        auto const qf = dx::round(
            val * fmath::inv_ln2, rounding::no_exc | rounding::to_nearest_int);
        auto const q = dx::cast<sint>(qf);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.5f,
            0.166666671633720397949219f,   //
            0.0416664853692054748535156f,  //
            0.00833336077630519866943359f, //
            0.00139304355252534151077271f>
            polynomial;
        auto u = dx::fmadd(dx::multiply(s, s), polynomial(s), s) + dx::one;
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        if constexpr (brain_float<E>) {
            u = dx::select(val > 100.0, dx::infinity, u);
            // underflow
            return dx::bit_drop(val < -92.186785, u);
        } else {
            static_assert(digits_v<E> == 12 && sizeof(E) == 2);
            static constexpr E max_ln = 11.089866;
            static constexpr E min_ln = -16.63553;
            u = dx::select(val > max_ln, dx::infinity, u);
            // underflow
            return dx::bit_drop(val < min_ln, u);
        }
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        using sint = signed_representation_t<float>;
        auto const qf = dx::round(
            val * fmath::inv_ln2, rounding::no_exc | rounding::to_nearest_int);
        auto const q = dx::cast<sint>(qf);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.5f,
            0.166666671633720397949219f,   //
            0.0416664853692054748535156f,  //
            0.00833336077630519866943359f, //
            0.00139304355252534151077271f, //
            0.000198527617612853646278381f>
            polynomial;
        // x2 * f + x + 1
        auto u = dx::fmadd(dx::multiply(s, s), polynomial(s), s) + dx::one;
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val > 100.0f, dx::infinity, u);
        // underflow
        return dx::bit_drop(val < -103.97208f, u);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<double, A> val) noexcept {
        using sint = signed_representation_t<double>;
        auto const q = dx::cast<sint>(val * fmath::inv_ln2);
        auto const qf = dx::cast<double>(q);
        using fpair = fmath::pair<double, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.5, 0.1666666666666669072e+0,
            0.4166666666666602598e-1, 0.8333333333314938210e-2,
            0.1388888888914497797e-2, 0.1984126989855865850e-3,
            0.2480158687479686264e-4, 0.2755723402025388239e-5,
            0.2755762628169491192e-6, 0.2511210703042288022e-7,
            0.2081276378237164457e-8>
            polynomial;
        auto u = dx::fmadd(dx::multiply(s, s), polynomial(s), s) + dx::one;
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        static constexpr auto max_log = 0x1.62e42fefa39efp+9;
        u = dx::select(val > max_log, dx::infinity, u);
        // underflow
        return dx::bit_drop(val < -745.133, u);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              exp(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return exp(internal::abi<T>, val);
                } else {
                    return fallback(val);
                }
            } else {
                return exp(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::exp_t exp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

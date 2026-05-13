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
void exp2(...) noexcept = delete;

template <typename T>
concept unqualified_exp2 = floating_point_simd<T> && requires(T val) {
    { exp2(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

struct exp2_t {
private:
    template <floating_point E, simd_abi A>
    requires (dx::digits_v<E> < dx::digits_v<float>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        using sint = signed_rep_t<float>;
        auto const qf = dx::round(
            val * fmath::inv_ln2, rounding::no_exc | rounding::to_nearest_int);
        auto const q = dx::cast<sint>(qf);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.6931471825f, //
            0.2402264476f,                                //
            0.5550347269e-1f,                             //
            0.9618384764e-2f>
            polynomial;
        static constexpr E max_log = 11;
        auto u = dx::fmadd(polynomial(s), s, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        if constexpr (brain_float<E>) {
            u = dx::select(val >= 128.0f, dx::infinity, u);
            // underflow
            return dx::bit_keep(val >= -133.0, u);
        } else {
            static_assert(digits_v<E> == 12 && sizeof(E) == 2);
            u = dx::select(val >= 16.0, dx::infinity, u);
            // underflow
            return dx::bit_keep(val >= -24.0, u);
        }
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        using sint = signed_rep_t<float>;
        auto const qf = dx::round(
            val * fmath::inv_ln2, rounding::no_exc | rounding::to_nearest_int);
        auto const q = dx::cast<sint>(qf);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.6931471825f, //
            0.2402264476f,                                //
            0.5550347269e-1f,                             //
            0.9618384764e-2f,                             //
            0.1339262701e-2f,                             //
            0.1535920892e-3f>
            polynomial;
        auto u = dx::fmadd(polynomial(s), s, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val >= 128.0f, dx::infinity, u);
        // underflow
        return dx::bit_keep(val >= -149.0f, u);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<double, A> val) noexcept {
        using sint = signed_rep_t<double>;
        auto const q = dx::cast<sint>(val * fmath::inv_ln2);
        auto const qf = dx::cast<double>(q);
        using fpair = fmath::pair<double, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.6931471805599452862,
            0.2402265069591012214e+0, 0.5550410866482046596e-1,
            0.9618129107597600536e-2, 0.1333355814670499073e-2,
            0.1540353045101147808e-3, 0.1525273353517584730e-4,
            0.1321543872511327615e-5, 0.1017819260921760451e-6,
            0.7073164598085707425e-8, 0.4434359082926529454e-9>
            polynomial;
        auto u = dx::fmadd(polynomial(s), s, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val >= 1024.0, dx::infinity, u);
        // underflow
        return dx::bit_keep(val >= -1074.0, u);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              exp2(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return exp2(internal::abi<T>, val);
                } else {
                    return fallback(val);
                }
            } else {
                return exp2(internal::abi<T>, val);
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
DPL_EXPORT inline constexpr internal::exp2_t exp2{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

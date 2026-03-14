// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/floating_point_simd_with_abi.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void pow(...) noexcept = delete;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_pow = requires(L lhs, R rhs) {
    { pow(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};

struct pow_t : binary_operation_base<pow_t> {
private:
    friend binary_operation_base<pow_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        {
            pow(internal::abi<A>, left, right)
        } -> floating_point_simd_with_abi<A>;
    }
    {
        return pow(internal::abi<A>, left, right);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<float, A> DPL_VECTORCALL
        exp2(fmath::pair<float, A> arg) noexcept {
        // A little more expensive than dx::exp2 but results in
        // better precision for this use-case
        using fpair = fmath::pair<float, A>;

        auto u = arg.upper + arg.lower;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        auto const q = dx::cast<signed_rep_t<float>>(qf);
        auto s = fmath::normalize(arg - qf);
        // polynomial for f(x) = (pow(2,x) - 1 - x ln(2)) / pow(x,2)
        static constexpr fmath::polynomial<0.24022650718688965f, //
            0.055503811687231064f,                               //
            0.00961806159466505f,                                //
            0.0013381305616348982f,                              //
            0.0001546145067550242f>
            polynomial;
        u = polynomial(s.upper);
        constexpr auto one = dx::one_v<basic_simd<float, A>>;

        // t = pow(2,x) where x is in the interval [-0.5,0.5]
        // |s| <= 0.5
        // |u| < 0.271
        // ln2 ~ 0.69
        // so assumptions for fast arithmetic holds
        auto t = fmath::fast(one) +
            (fmath::fast(fmath::ln2_v<fpair> * s) + fmath::square(s) * u);
        // zero if underflow
        return dx::bit_drop(arg.upper < -150.0f,
            fmath::ldexp(fmath::compliance::speed, t.upper + t.lower, q));
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<double, A> DPL_VECTORCALL
        exp2(fmath::pair<double, A> arg) noexcept {
        // A little more expensive than dx::exp2 but results in
        // better precision for this use-case
        using fpair = fmath::pair<double, A>;

        auto u = arg.upper + arg.lower;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        auto const q = dx::cast<signed_rep_t<double>>(qf);
        auto s = fmath::normalize(arg - qf);
        // polynomial for f(x) = (pow(2,x) - 1 - x ln(2)) / pow(x,2)
        static constexpr fmath::polynomial<0.24069579622573783,
            0.06440213344186142, 0.087906020124066, 0.3927994222793298,
            1.1138080524555194, 1.399782605545827, -1.3239033234264652,
            -7.953882600802256, -12.875978831585796, -9.890373910625978,
            -3.065528692252689>
            polynomial;
        u = polynomial(s.upper);
        constexpr auto one = dx::one_v<basic_simd<double, A>>;

        // t = pow(2,x) where x is in the interval [-0.5,0.5]
        // |s| <= 0.5
        // |u| < 0.271
        // ln2 ~ 0.69
        // so assumptions for fast arithmetic holds
        auto t = fmath::fast(one) +
            (fmath::fast(fmath::ln2_v<fpair> * s) + fmath::square(s) * u);
        // zero if underflow
        return dx::bit_drop(arg.upper < -1000.0f,
            fmath::ldexp(fmath::compliance::speed, t.upper + t.lower, q));
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<float, A> DPL_VECTORCALL
        log2(basic_simd<float, A> arg) noexcept {
        // takes a decomposed significand; only valid in interval [0.75,1.5)

        constexpr auto n_one = fmath::single(dx::broadcast<A>(-1.0f));
        constexpr auto one = fmath::single(dx::broadcast<A>(1.0f));

        auto const x = (n_one + arg) / (one + arg);
        auto const x2 = fmath::square(x);
        // polynomial for atanh
        constexpr fmath::polynomial<0.400007992982864379882812f, //
            0.285112679004669189453125f,                         //
            0.240320354700088500976562f>
            polynomial;
        auto t = polynomial(x2.upper);
        constexpr fmath::pair<float, A> onethird = fmath::make_pair<A>(
            0.66666662693023681640625f, 3.69183861259614332084311e-09f);

        auto s = fmath::scale(x, dx::broadcast<A>(2.0f));
        s = fmath::fast(s) + (x2 * x * (x2 * t + onethird));
        constexpr fmath::pair<float, A> inv_ln2 = fmath::make_pair<A>(
            1.44269502162933349609f, 1.92596299112661746887e-08f);
        return s * inv_ln2;
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<double, A> DPL_VECTORCALL
        log2(basic_simd<double, A> arg) noexcept {
        // takes a decomposed significand; only valid in interval [0.75,1.5)

        constexpr auto n_one = fmath::single(dx::broadcast<A>(-1.0));
        constexpr auto one = fmath::single(dx::broadcast<A>(1.0));

        auto const x = (n_one + arg) / (one + arg);
        auto const x2 = fmath::square(x);
        // polynomial for atanh
        constexpr fmath::polynomial<0.400000000000000077715612,
            0.285714285714249172087875, 0.222222222230083560345903,
            0.181818180850050775676507, 0.153846227114512262845736,
            0.13332981086846273921509, 0.117754809412463995466069,
            0.103239680901072952701192, 0.116255524079935043668677>
            polynomial;
        auto t = polynomial(x2.upper);
        constexpr fmath::pair<double, A> onethird = fmath::make_pair<A>(
            0.666666666666666629659233, 3.80554962542412056336616e-17);
        auto s = fmath::scale(x, dx::broadcast<A>(2.0));
        s = fmath::fast(s) + (x2 * x * (x2 * t + onethird));
        constexpr fmath::pair<double, A> inv_ln2 =
            fmath::make_pair<A>(1.44269504088896338700465091244,
                2.03552737684314300702482381274e-17);
        return s * inv_ln2;
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> lhs, basic_simd<E, A> rhs) noexcept {
        auto const absl = dx::abs(lhs);
        auto const [fr, exp] =
            dx::frexp(absl, frexp_reduced | frexp_floating_point);
        auto result = pow_t::exp2(rhs * (pow_t::log2(fr) + exp));

        constexpr auto inf = dx::infinity_v<decltype(result)>;

        auto const efx = dx::fixup(dx::sign(absl - dx::one, rhs), inf,
            fpfix::condition<fpfix::negative, dx::zero> |
                fpfix::condition<fpfix::zero, dx::one>);

        result = dx::select(dx::isinf(rhs), efx, result);

        auto const islhs_zero = lhs == dx::zero;
        constexpr auto is_odd = [](auto rhs) {
            using sint = signed_rep_t<E>;
            return (dx::cast<sint>(rhs) & dx::one) == dx::one &&
                dx::trunc(rhs) == rhs && dx::abs(rhs) < fmath::maxint;
        };

        result = dx::select(dx::isinf(lhs) || islhs_zero,
            dx::negate(is_odd(rhs) && lhs < dx::zero,
                dx::bit_drop(dx::signbit(rhs) ^ islhs_zero, dx::infinity)),
            result);
        result = dx::bit_fill(dx::isnan(lhs) || dx::isnan(rhs), result);

        return dx::select(rhs == dx::zero || lhs == dx::one, dx::one, result);
    }

public:
    template <floating_point_simd L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_pow<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return pow(internal::abi<A>, lhs, rhs);
                }
            } else {
                return pow(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <floating_point_simd L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_pow<L, R> ||
            unqualified_pow<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_pow<L, R>) {
            return pow(internal::abi<A>, lhs, rhs);
        } else {
            return pow(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<pow_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::pow_t pow{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

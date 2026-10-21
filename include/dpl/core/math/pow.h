// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/exp2.h"
#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/infinity.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/internal/transform.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/std/concepts/tuple_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void pow(...) noexcept = delete;

struct DPL_EMPTY_BASES pow_t :
    public math_operation_base<pow_t>,
    public maskable_transform_base<pow_t> {
    using math_operation_base<pow_t>::operator();
    using maskable_transform_base<pow_t>::operator();
};

template <>
struct operation_signature<pow_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_pow = requires {
    {
        pow(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename T>
concept unqualified_canonical_mpow_base =
    cpo_invocable<pow_t, L, R> && requires {
        {
            pow(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<T>;
    };

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_mpow =
    unqualified_canonical_mpow_base<T, M, L, R, T>;

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_zmpow =
    unqualified_canonical_mpow_base<dx::zero_t, M, L, R, T>;

template <>
struct canonical_impl<pow_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<pow_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, same_as<L> R>
    requires canonical_vector<R> && unqualified_canonical_pow<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        return pow(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_pow<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return pow(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_pow<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return pow(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_terminal_of<pow_t, L> R>
    requires unqualified_canonical_mpow<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return pow(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_mpow<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return pow(internal::abi<T>, src, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_terminal_of<pow_t, L> R>
    requires unqualified_canonical_zmpow<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return pow(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_zmpow<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return pow(internal::abi<T>, zero, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename L, typename R, typename T>
concept unqualified_extended_pow = requires {
    {
        pow(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mpow_base =
    cpo_invocable<pow_t, L, R> && requires {
        {
            pow(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<pow_t, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mpow =
    equivalent_vector_with<S, cpo_result_t<pow_t, L, R>> &&
    unqualified_extended_mpow_base<S, M, L, R>;

template <typename M, typename L, typename R>
concept unqualified_extended_zmpow =
    unqualified_extended_mpow_base<dx::zero_t, M, L, R>;

template <>
struct extended_impl<pow_t> {
public:
    template <simd_vector L, equivalent_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_pow<L, R, common_canonical_simd_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return pow(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_pow<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return pow(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_pow<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return pow(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mpow<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return pow(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mpow<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return pow(src, dx::to_const_mask<S>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_mask_for<pow_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmpow<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return pow(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<pow_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmpow<
            launder_cmask_t<cpo_result_t<pow_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return pow(zero, dx::to_const_mask<cpo_result_t<pow_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<pow_t> : binary_canonical_broadcaster<pow_t> {
private:
    template <mx::simd_double T>
    requires same_as<simd_value_type_t<T>, float> ||
        same_as<simd_value_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL exp2(T arg) noexcept {
        // A little more expensive than dx::exp2 but results in
        // better precision for this use-case
        using E = simd_value_type_t<T>;
        using A = simd_abi_type_t<T>;
        using exp2_impl = fallback_impl<exp2_t>;

        constexpr auto self = [](T arg, simd_element_type_t<T> varg) {
            auto const qf = dx::roundeven(varg, rounding::no_exc);
            auto s = mx::normalize(mx::pair_ref{arg} - qf);
            auto const u = exp2_impl::solve_poly(dx::get_element<0>(s));
            // t = pow(2,x) where x is in the interval [-0.5,0.5]
            // |s| <= 0.5
            // |u| < 1
            // so s * u < 1
            // so assumptions for fast arithmetic holds
            s = mx::pair_ref{s} * u;
            s = mx::fast(dx::broadcast<E, A>(dx::one)) + s;
            // zero if underflow
            return dx::select(dx::cmplt(varg, exp2_impl::lower_bound<E>),
                dx::zero,
                mx::ldexp(mx::compliance::speed, mx::recombine(s), qf));
        };

        if consteval {
            auto const varg = mx::recombine(arg);
            auto const oob = dx::cmpge(varg, exp2_impl::upper_bound<E>);
            auto const result =
                self(mx::select(oob, mx::make_pair<A, E>(dx::zero), arg),
                    dx::select(oob, dx::zero, varg));
            return dx::select(oob, dx::infinity, result);
        } else {
            return self(arg, mx::recombine(arg));
        }
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL log2_poly(T arg) noexcept {
        // polynomial for atanh
        using E = simd_element_type_t<T>;
        if constexpr (is_same_v<E, double>) {
            mx::polynomial<0.400000000000000077715612,
                0.285714285714249172087875, 0.222222222230083560345903,
                0.181818180850050775676507, 0.153846227114512262845736,
                0.13332981086846273921509, 0.117754809412463995466069,
                0.103239680901072952701192, 0.116255524079935043668677>
                poly;
            return poly(arg);
        } else {
            constexpr mx::polynomial<0.400007992982864379882812f, //
                0.285112679004669189453125f,                      //
                0.240320354700088500976562f>
                poly;
            return poly(arg);
        }
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr mx::pair_type_t<T>
        DPL_VECTORCALL log2(T arg) noexcept {

        // takes a decomposed significand; only valid in interval [0.75,1.5)
        auto const vone = dx::broadcast<T>(dx::one);
        auto const x =
            mx::pair_ref{arg - mx::single{vone}} / (mx::single{vone} + arg);
        auto const x2 = mx::square(x);
        auto t = log2_poly(dx::get_element<0>(x2));

        using E = simd_element_type_t<T>;
        auto const twothirds = []() {
            if constexpr (is_same_v<E, double>) {
                return mx::make_pair( //
                    dx::broadcast<T>(0x1.5555555555555p-1),
                    dx::broadcast<T>(0x1.5555555555555p-55));
            } else {
                return mx::make_pair( //
                    dx::broadcast<T>(0x1.555556p-1f),
                    dx::broadcast<T>(-0x1.555556p-26f));
            }
        }();

        auto const x3 = mx::pair_ref{x2} * x;
        auto s = mx::pair_ref{x2} * t;
        s = mx::pair_ref{s} + twothirds;
        s = mx::pair_ref{x3} * s;
        s = mx::fast(mx::scale(x, 2.0)) + s;
        auto const inv_ln2 = []() {
            if constexpr (is_same_v<E, double>) {
                return mx::make_pair( //
                    dx::broadcast<T>(0x1.71547652b82fep+0),
                    dx::broadcast<T>(0x1.777d0ffda0d24p-56));
            } else {
                return mx::make_pair( //
                    dx::broadcast<T>(0x1.715476p+0f),
                    dx::broadcast<T>(0x1.4ae0cp-26f));
            }
        }();

        return mx::pair_ref{s} * inv_ln2;
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL pow(T lhs, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
        constexpr auto impl = [](T absl, T rhs) {
            if constexpr (is_same_v<E, float> || is_same_v<E, double>) {
                auto const [fr, exp] =
                    dx::to_tuple_like(dx::frexp(absl, frexp_options::reduced));
                return fallback_impl::exp2(rhs *
                    mx::pair_ref{mx::pair_ref{fallback_impl::log2(fr)} + exp});
            } else {
                auto const powf = [](auto lhs, auto rhs) {
                    auto const [fr, exp] = dx::to_tuple_like(
                        dx::frexp(lhs, frexp_options::reduced));
                    return dx::exp2(
                        dx::multiply(dx::add(dx::log2(fr), exp), rhs));
                };

                return mx::apply_widened<float>(powf, absl, rhs);
            }
        };
        auto const absl = dx::abs(lhs);
        auto result = [&] {
            if consteval {
                auto const cleanl =
                    dx::select(dx::isfinite(absl), absl, dx::zero);
                auto const cleanr =
                    dx::select(dx::isfinite(rhs), rhs, dx::zero);
                return impl(cleanl, cleanr);
            } else {
                return impl(absl, rhs);
            }
        }();

        auto const isrhs_odd = [](auto val) {
            constexpr E half = 0.5;
            if consteval {
                val = dx::select(dx::isfinite(val), val, dx::zero);
            }

            auto const h = dx::multiply(val, half); // y/2, exact
            return dx::cmpeq(dx::abs(dx::subtract(h, dx::trunc(h))), half);
        }(rhs);

        auto const islhs_ltz = dx::signbit(lhs);
        {
            auto const isrhs_int = dx::cmpeq(dx::trunc(rhs), rhs);
            result = dx::select(                    //
                dx::bwandnot(islhs_ltz, isrhs_int), //
                dx::all_bits,                       //
                dx::negate(
                    result, dx::logical_and(isrhs_odd, islhs_ltz), result));
        }

        auto const vinf = dx::broadcast<T>(dx::infinity);

        {
            auto efx = [&] {
                if consteval {
                    auto const isnan = dx::isnan(absl);
                    return dx::select(isnan, absl,
                        dx::subtract(
                            dx::select(isnan, dx::zero, absl), dx::one));
                } else {
                    return dx::subtract(absl, dx::one);
                }
            }();
            efx = dx::sign(efx, rhs);
            auto const ltz = dx::cmplt(efx, dx::zero);
            auto const eqz = dx::cmpeq(efx, dx::zero);
            efx = dx::select(ltz, dx::zero, dx::select(eqz, dx::one, vinf));
            result = dx::select(dx::isinf(rhs), efx, result);
        }

        auto const islhs_zero = dx::cmpeq(lhs, dx::zero);
        auto const invalid = dx::select(
            dx::cmpneq(dx::signbit(rhs), islhs_zero), dx::zero, vinf);
        auto const negated =
            dx::negate(invalid, dx::logical_and(isrhs_odd, islhs_ltz), invalid);
        result = dx::select(
            dx::logical_or(dx::isinf(lhs), islhs_zero), negated, result);

        result = dx::select(dx::cmpunord(lhs, rhs), dx::all_bits, result);

        return dx::select(
            dx::logical_or(dx::cmpeq(rhs, dx::zero), dx::cmpeq(lhs, dx::one)),
            dx::one, result);
    }

public:
    template <canonical_vector T, same_as<T> U>
    requires cpo_invocable<frexp_t, T, frexp_options::reduced_t> &&
        (same_as<simd_element_type_t<T>, double> ||
            same_as<simd_element_type_t<T>, float> ||
            same_as<simd_element_type_t<T>, ext::float16> ||
            same_as<simd_element_type_t<T>, ext::bfloat16>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T lhs, U rhs) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_pow)
        if consteval {
            if constexpr (fixed_width_abi<A>) {
                return internal::transform<T>(
                    [](E lhs, E rhs) {
                        if constexpr (is_same_v<E, double>) {
                            return __builtin_pow(lhs, rhs);
                        } else {
                            return __builtin_powf(static_cast<float>(lhs),
                                static_cast<float>(rhs));
                        }
                    },
                    lhs, rhs);
            }
        }
#endif
        return fallback_impl::pow(lhs, rhs);
    }

    using binary_canonical_broadcaster<pow_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::pow_t pow{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

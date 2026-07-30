// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
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
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/std/concepts/tuple_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void pow(...) noexcept = delete;

struct DPL_EMPTY_BASES pow_t :
    private math_operation_base<pow_t>,
    private maskable_transform_base<pow_t> {
    using math_operation_base<pow_t>::operator();
    using maskable_transform_base<pow_t>::operator();
};

template <>
struct operation_signature<pow_t> {
    static consteval void operator()(
        simd_vector auto&&, simd_vector auto&&) noexcept {}
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_pow = requires {
    {
        pow(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<basic_vector<simd_element_type_t<L>, A>>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_mpow = cpo_invocable<pow_t, L, R> &&
    (!simd_type<S> || same_as<S, cpo_result_t<pow_t, L, R>>) && requires {
        {
            pow(internal::abi<cpo_result_t<pow_t, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<pow_t, L, R>>;
    };

template <>
struct canonical_impl<pow_t> {
private:
    template <typename L, typename R>
    using vresult_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using vmask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R, typename M>
    using vcmask_t DPL_NODEBUG = launder_cmask_t<cpo_result_t<pow_t, L, R>, M>;

public:
    template <canonical_vector L, canonical_vector R>
    requires canonical_vector<R> && unqualified_canonical_pow<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(L lhs, R rhs) noexcept {
        return pow(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R>
    requires canonical_vector<R> &&
        unqualified_canonical_mpow<vresult_t<L, R>, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return pow(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R,
        const_mask_for<vresult_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mpow<vresult_t<L, R>, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return pow(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R>
    requires canonical_vector<R> &&
        unqualified_canonical_mpow<dx::zero_t, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return pow(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R,
        const_mask_for<vresult_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mpow<dx::zero_t, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return pow(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_pow = requires {
    {
        pow(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mpow = cpo_invocable<pow_t, L, R> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<pow_t, L, R>>) &&
    requires {
        {
            pow(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<pow_t, L, R>>;
    };

template <>
struct extended_impl<pow_t> {
private:
    template <typename L, typename R, typename M>
    using vcmask_t DPL_NODEBUG = launder_cmask_t<cpo_result_t<pow_t, L, R>, M>;

public:
    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_pow<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return pow(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mpow<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return pow(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mpow<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return pow(src, dx::to_const_mask<S>(cmask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, simd_vector R, result_mask_for<pow_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mpow<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return pow(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, simd_vector R, result_cmask_for<pow_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mpow<dx::zero_t, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return pow(zero, dx::to_const_mask<cpo_result_t<pow_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<pow_t> {
private:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A>
        DPL_VECTORCALL exp2(fmath::pair<float, A> arg) noexcept {
        // A little more expensive than dx::exp2 but results in
        // better precision for this use-case
        using fpair = fmath::pair<float, A>;
        using sint = signed_representation_t<float>;

        auto u = arg.upper + arg.lower;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        auto const q = dx::element_cast<sint>(qf);
        auto s = fmath::normalize(arg - qf);
        // polynomial for f(x) = (pow(2,x) - 1 - x ln(2)) / pow(x,2)
        static constexpr fmath::polynomial<0.24022650718688965f, //
            0.055503811687231064f,                               //
            0.00961806159466505f,                                //
            0.0013381305616348982f,                              //
            0.0001546145067550242f>
            polynomial;
        u = polynomial(s.upper);
        auto const one = dx::broadcast<float, A>(dx::one);

        // t = pow(2,x) where x is in the interval [-0.5,0.5]
        // |s| <= 0.5
        // |u| < 0.271
        // ln2 ~ 0.69
        // so assumptions for fast arithmetic holds
        auto t = fmath::fast(one) +
            (fmath::fast(fmath::ln2_v<fpair> * s) + fmath::square(s) * u);
        // zero if underflow
        return dx::select(arg.upper < -150.0f, dx::zero,
            fmath::ldexp(fmath::compliance::speed, t.upper + t.lower, q));
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<double, A>
        DPL_VECTORCALL exp2(fmath::pair<double, A> arg) noexcept {
        // A little more expensive than dx::exp2 but results in
        // better precision for this use-case
        using fpair = fmath::pair<double, A>;
        using sint = signed_representation_t<double>;

        auto u = arg.upper + arg.lower;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        auto const q = dx::element_cast<sint>(qf);
        auto s = fmath::normalize(arg - qf);
        // polynomial for f(x) = (pow(2,x) - 1 - x ln(2)) / pow(x,2)
        constexpr fmath::polynomial<0.24069579622573783, 0.06440213344186142,
            0.087906020124066, 0.3927994222793298, 1.1138080524555194,
            1.399782605545827, -1.3239033234264652, -7.953882600802256,
            -12.875978831585796, -9.890373910625978, -3.065528692252689>
            polynomial;
        u = polynomial(s.upper);
        auto const one = dx::broadcast<double, A>(dx::one);

        // t = pow(2,x) where x is in the interval [-0.5,0.5]
        // |s| <= 0.5
        // |u| < 0.271
        // ln2 ~ 0.69
        // so assumptions for fast arithmetic holds
        auto t = fmath::fast(one) +
            (fmath::fast(fmath::ln2_v<fpair> * s) + fmath::square(s) * u);
        // zero if underflow
        return dx::select(arg.upper < -1000.0f, dx::zero,
            fmath::ldexp(fmath::compliance::speed, t.upper + t.lower, q));
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<float, A>
        DPL_VECTORCALL log2(basic_vector<float, A> arg) noexcept {
        // takes a decomposed significand; only valid in interval [0.75,1.5)
        auto const one = fmath::single(dx::broadcast<float, A>(dx::one));
        auto const x = (arg - one) / (one + arg);
        auto const x2 = fmath::square(x);
        // polynomial for atanh
        constexpr fmath::polynomial<0.400007992982864379882812f, //
            0.285112679004669189453125f,                         //
            0.240320354700088500976562f>
            polynomial;
        auto t = polynomial(x2.upper);
        fmath::pair<float, A> const onethird = fmath::make_pair<A>(
            0.66666662693023681640625f, 3.69183861259614332084311e-09f);

        auto s = fmath::scale(x, dx::broadcast<A>(2.0f));
        s = fmath::fast(s) + (x2 * x * (x2 * t + onethird));
        fmath::pair<float, A> const inv_ln2 = fmath::make_pair<A>(
            1.44269502162933349609f, 1.92596299112661746887e-08f);
        return s * inv_ln2;
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<double, A>
        DPL_VECTORCALL log2(basic_vector<double, A> arg) noexcept {
        // takes a decomposed significand; only valid in interval [0.75,1.5)
        auto const one = fmath::single(dx::broadcast<double, A>(dx::one));
        auto const x = (arg - one) / (one + arg);
        auto const x2 = fmath::square(x);
        // polynomial for atanh
        constexpr fmath::polynomial<0.400000000000000077715612,
            0.285714285714249172087875, 0.222222222230083560345903,
            0.181818180850050775676507, 0.153846227114512262845736,
            0.13332981086846273921509, 0.117754809412463995466069,
            0.103239680901072952701192, 0.116255524079935043668677>
            polynomial;
        auto t = polynomial(x2.upper);
        fmath::pair<double, A> const onethird = fmath::make_pair<A>(
            0.666666666666666629659233, 3.80554962542412056336616e-17);
        auto s = fmath::scale(x, dx::broadcast<A>(2.0));
        s = fmath::fast(s) + (x2 * x * (x2 * t + onethird));
        fmath::pair<double, A> const inv_ln2 =
            fmath::make_pair<A>(1.44269504088896338700465091244,
                2.03552737684314300702482381274e-17);
        return s * inv_ln2;
    }

public:
    template <canonical_vector T>
    requires cpo_invocable<frexp_t, T, frexp_options::reduced_t> &&
        requires(
            T rhs, cpo_result_t<frexp_t, T, frexp_options::reduced_t> result) {
            fallback_impl::exp2(
                rhs * fallback_impl::log2(ranges::get_element<0>(result)) +
                ranges::get_element<1>(result));
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T lhs, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;

        auto const absl = dx::abs(lhs);
        auto const [fr, exp] =
            dx::to_tuple_like(dx::frexp(absl, frexp_options::reduced));
        auto result =
            fallback_impl::exp2(rhs * (fallback_impl::log2(fr) + exp));

        auto const inf = dx::broadcast<E, A>(dx::infinity);

        auto const efx = dx::fixup(dx::sign(absl - dx::one, rhs), inf,
            fpfix::condition<fpfix::negative, dx::zero> |
                fpfix::condition<fpfix::zero, dx::one>);

        result = dx::select(dx::isinf(rhs), efx, result);

        auto const islhs_zero = lhs == dx::zero;
        constexpr auto is_odd = [](auto rhs) {
            using sint = signed_representation_t<E>;
            return (dx::element_cast<sint>(rhs) & dx::one) == dx::one &&
                dx::trunc(rhs) == rhs && dx::abs(rhs) < fmath::maxint;
        };

        auto const invalid =
            dx::select(dx::signbit(rhs) ^ islhs_zero, dx::zero, inf);
        auto const negated =
            dx::negate(invalid, is_odd(rhs) && lhs < dx::zero, invalid);
        result = dx::select(dx::isinf(lhs) || islhs_zero, negated, result);
        result =
            dx::select(dx::isnan(lhs) || dx::isnan(rhs), dx::all_bits, result);

        return dx::select(rhs == dx::zero || lhs == dx::one, dx::one, result);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::pow_t pow{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

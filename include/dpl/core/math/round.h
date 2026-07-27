// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/ceil.h"
#include "dpl/core/math/copysign.h"
#include "dpl/core/math/details/constants.h"
#include "dpl/core/math/floor.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/rounding.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwandnot.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES round_t :
    private math_operation_base<round_t>,
    private maskable_transform_base<round_t> {
    using math_operation_base<round_t>::operator();
    using maskable_transform_base<round_t>::operator();
};

template <>
struct operation_signature<round_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}

    template <mx::rounding_flags R>
    static consteval void operator()(
        simd_vector auto&&, mx::rounding_t<R>) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mcround = cpo_invocable<round_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<round_t, T>>) && requires {
        {
            round(internal::abi<cpo_result_t<round_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<round_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_canonical_mcroundne =
    cpo_invocable<round_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        same_as<S, cpo_result_t<round_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<round_t, T, rounding::no_exc_t>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), rounding::no_exc)
        } -> same_as<cpo_result_t<round_t, T, rounding::no_exc_t>>;
    };

template <typename S, typename M, typename T, mx::rounding_flags R>
concept unqualified_canonical_mround =
    cpo_invocable<round_t, T, mx::rounding_t<R>> &&
    (!simd_type<S> ||
        same_as<S, cpo_result_t<round_t, T, mx::rounding_t<R>>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<round_t, T, mx::rounding_t<R>>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), mx::rounding_v<R>)
        } -> same_as<cpo_result_t<round_t, T, mx::rounding_t<R>>>;
    };

template <>
struct canonical_impl<round_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { round(internal::abi<A>, val); }
    {
        return round(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcround<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcround<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcround<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcround<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }

    ///
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept
    requires requires { round(internal::abi<A>, val, rounding::no_exc); }
    {
        return round(internal::abi<A>, val, rounding::no_exc);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcroundne<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, mask_t<T> mask, T val,
        rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, rounding::no_exc);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcroundne<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            rounding::no_exc);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcroundne<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, mask, val, rounding::no_exc);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcroundne<dx::zero_t, launder_cmask_t<T, M>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            rounding::no_exc);
    }

    ///
    template <simd_abi A, simd_element_for<A> E, mx::rounding_flags R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, mx::rounding_t<R> flags) noexcept
    requires requires { round(internal::abi<A>, val, flags); }
    {
        return round(internal::abi<A>, val, flags);
    }

    template <canonical_vector T, mx::rounding_flags R>
    requires unqualified_canonical_mround<type_identity_t<T>, mask_t<T>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, mask_t<T> mask, T val,
        mx::rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, src, mask, val, flags);
    }

    template <canonical_vector T, const_mask_for<T> M, mx::rounding_flags R>
    requires unqualified_canonical_mround<type_identity_t<T>,
        launder_cmask_t<T, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, M cmask, T val,
        mx::rounding_t<R> flags) noexcept {
        return round(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, flags);
    }

    template <canonical_vector T, mx::rounding_flags R>
    requires unqualified_canonical_mround<dx::zero_t, mask_t<T>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, mask_t<T> mask, T val,
        mx::rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, zero, mask, val, flags);
    }

    template <canonical_vector T, const_mask_for<T> M, mx::rounding_flags R>
    requires unqualified_canonical_mround<dx::zero_t, launder_cmask_t<T, M>, T,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, mx::rounding_t<R> flags) noexcept {
        return round(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, flags);
    }
};

template <typename T>
concept unqualified_extended_cround = requires(T val) {
    { round(val) } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept unqualified_extended_croundne = requires(T val) {
    {
        round(val, rounding::no_exc)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};
template <typename T, mx::rounding_flags R>
concept unqualified_extended_round = requires(T val) {
    {
        round(val, mx::rounding_v<R>)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mcround = cpo_invocable<round_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<round_t, T>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<round_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mcroundne =
    cpo_invocable<round_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        equivalent_vector_with<S,
            cpo_result_t<round_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), rounding::no_exc)
        }
        -> equivalent_vector_with<cpo_result_t<round_t, T, rounding::no_exc_t>>;
    };

template <typename S, typename M, typename T, mx::rounding_flags R>
concept unqualified_extended_mround =
    cpo_invocable<round_t, T, mx::rounding_t<R>> &&
    (!simd_type<S> ||
        equivalent_vector_with<S,
            cpo_result_t<round_t, T, mx::rounding_t<R>>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), mx::rounding_v<R>)
        }
        -> equivalent_vector_with<cpo_result_t<round_t, T, mx::rounding_t<R>>>;
    };

template <>
struct extended_impl<round_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_cround<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return round(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcround<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcround<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<round_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcround<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<round_t, T> M>
    requires unqualified_extended_mcround<dx::zero_t,
        launder_cmask_t<cpo_result_t<round_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return round(zero, dx::to_const_mask<cpo_result_t<round_t, T>>(cmask),
            __DPL forward<T>(val));
    }
    ///
    template <extended_vector T>
    requires unqualified_extended_croundne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding::no_exc_t) {
        return round(__DPL forward<T>(val), rounding::no_exc);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcroundne<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), rounding::no_exc);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcroundne<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), rounding::no_exc);
    }

    template <simd_vector T, result_mask_for<round_t, T, rounding::no_exc_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcroundne<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, rounding::no_exc_t) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            rounding::no_exc);
    }

    template <extended_vector T,
        result_cmask_for<round_t, T, rounding::no_exc_t> M>
    requires unqualified_extended_mcroundne<dx::zero_t,
        launder_cmask_t<cpo_result_t<round_t, T, rounding::no_exc_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<round_t, T, rounding::no_exc_t>>(
                cmask),
            __DPL forward<T>(val), rounding::no_exc);
    }

    ///
    template <extended_vector T, mx::rounding_flags R>
    requires unqualified_extended_round<T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, mx::rounding_t<R> flags) {
        return round( __DPL forward<T>(val), flags);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        mx::rounding_flags R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mround<S, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, mx::rounding_t<R> flags) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), flags);
    }

    template <fixed_width_vector S, const_mask_for<S> M,
        common_vector_with<S> T, mx::rounding_flags R>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mround<S, launder_cmask_t<S, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, mx::rounding_t<R> flags) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), flags);
    }

    template <simd_vector T, mx::rounding_flags R,
        result_mask_for<round_t, T, mx::rounding_t<R>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mround<dx::zero_t, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, mx::rounding_t<R> flags) {
        return round(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), flags);
    }

    template <extended_vector T, mx::rounding_flags R,
        result_cmask_for<round_t, T, mx::rounding_t<R>> M>
    requires unqualified_extended_mround<dx::zero_t,
        launder_cmask_t<cpo_result_t<round_t, T, mx::rounding_t<R>>, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, mx::rounding_t<R> flags) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<round_t, T, mx::rounding_t<R>>>(
                cmask),
            __DPL forward<T>(val), flags);
    }
};

template <>
struct fallback_impl<round_t> {
    template <simd_abi A, simd_element_for<A> E>
    requires binary_layout_floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        // cmath rounding: round to nearest int, tie away from zero
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto x = finite + mx::half;
        auto fr = x - dx::trunc(x);
        auto const one = dx::broadcast<E, A>(dx::one);
        x = dx::subtract(x, x <= dx::zero && fr == dx::zero, x, one);
        fr = dx::add(fr, fr < dx::zero, fr, one);
        x = dx::select(fr != mx::underhalf, x, dx::zero);
        return dx::select(isfinite && dx::abs(val) < mx::maxint,
            dx::copysign(x - fr, finite), val);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires binary_layout_floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return operator()(val);
    }

    template <simd_abi A, simd_element_for<A> E, mx::rounding_flags R>
    requires binary_layout_floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> val, mx::rounding_t<R>) {
        constexpr auto opt = mx::rounding_v<R>;
        static_assert(opt);
        if constexpr (opt.has(rounding::to_zero | rounding::no_exc)) {
            return dx::trunc(val, rounding::no_exc);
        } else if constexpr (opt.has(rounding::to_zero)) {
            return dx::trunc(val);
        } else if constexpr (opt.has(rounding::to_pos_inf | rounding::no_exc)) {
            return dx::ceil(val, rounding::no_exc);
        } else if constexpr (opt.has(rounding::to_pos_inf)) {
            return dx::ceil(val);
        } else if constexpr (opt.has(rounding::to_neg_inf | rounding::no_exc)) {
            return dx::floor(val, rounding::no_exc);
        } else if constexpr (opt.has(rounding::to_neg_inf)) {
            return dx::floor(val);
        } else {
            // round to nearest int, tie to nearest even
            auto const isfinite = dx::isfinite(val);
            auto const finite = dx::select(isfinite, val, dx::zero);

            auto const floored = dx::floor(finite);
            auto const onef = dx::broadcast<E, A>(dx::one);
            auto fr = finite - floored;
            auto x = dx::add(finite, fr > 0.5f, finite, onef);

            // there are bit tricks alternatives to casting available but
            // they usually just add more instructions
            using sint = signed_representation_t<E>;
            auto const onei = dx::broadcast<sint, A>(dx::one);
            auto const isodd = (dx::element_cast<sint>(floored) & onei) == onei;
            auto const result =
                dx::copysign(dx::add(floored, isodd, floored, onef), finite);
            return dx::select(
                isfinite && dx::abs(val) < mx::maxint, result, val);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::round_t round{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

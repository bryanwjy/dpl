// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/ceil.h"
#include "dpl/core/math/copysign.h"
#include "dpl/core/math/floor.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwandnot.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
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

    template <rounding_flags R>
    static consteval void operator()(
        simd_vector auto&&, rounding_t<R>) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mcround =
    (!simd_type<S> || same_as<S, cpo_result_t<round_t, T>>) && requires {
        {
            round(internal::abi<cpo_result_t<round_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> same_as<cpo_result_t<round_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_canonical_mcroundne =
    (!simd_type<S> ||
        same_as<S, cpo_result_t<round_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<round_t, T, rounding::no_exc_t>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), rounding::no_exc)
        } -> same_as<cpo_result_t<round_t, T, rounding::no_exc_t>>;
    };

template <typename S, typename M, typename T, rounding_flags R>
concept unqualified_canonical_mround =
    (!simd_type<S> || same_as<S, cpo_result_t<round_t, T, rounding_t<R>>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<round_t, T, rounding_t<R>>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), rounding_v<R>)
        } -> same_as<cpo_result_t<round_t, T, rounding_t<R>>>;
    };

template <>
struct canonical_impl<round_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, M>;

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

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mcround<type_identity_t<T>, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, cmask_t<T, M> cmask, T val) noexcept {
        return round(internal::abi<T>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcround<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mcround<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return round(internal::abi<T>, zero, cmask, val);
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

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mcroundne<type_identity_t<T>, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, cmask_t<T, M> cmask,
        T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, cmask, val, rounding::no_exc);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcroundne<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, mask, val, rounding::no_exc);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mcroundne<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, cmask_t<T, M> cmask, T val,
        rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, cmask, val, rounding::no_exc);
    }

    ///
    template <simd_abi A, simd_element_for<A> E, rounding_flags R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding_t<R> flags) noexcept
    requires requires { round(internal::abi<A>, val, flags); }
    {
        return round(internal::abi<A>, val, flags);
    }

    template <canonical_vector T, rounding_flags R>
    requires unqualified_canonical_mround<type_identity_t<T>, mask_t<T>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, mask_t<T> mask, T val,
        rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, src, mask, val, flags);
    }

    template <fixed_width_vector T, imask_t<T> M, rounding_flags R>
    requires canonical_vector<T> &&
        unqualified_canonical_mround<type_identity_t<T>, cmask_t<T, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, cmask_t<T, M> cmask,
        T val, rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, src, cmask, val, flags);
    }

    template <canonical_vector T, rounding_flags R>
    requires unqualified_canonical_mround<dx::zero_t, mask_t<T>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, zero, mask, val, flags);
    }

    template <fixed_width_vector T, imask_t<T> M, rounding_flags R>
    requires canonical_vector<T> &&
        unqualified_canonical_mround<dx::zero_t, cmask_t<T, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, cmask_t<T, M> cmask, T val,
        rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, zero, cmask, val, flags);
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
template <typename T, rounding_flags R>
concept unqualified_extended_round = requires(T val) {
    { round(val, rounding_v<R>) } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mcround =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<round_t, T>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<round_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mcroundne =
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

template <typename S, typename M, typename T, rounding_flags R>
concept unqualified_extended_mround =
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<round_t, T, rounding_t<R>>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), rounding_v<R>)
        } -> equivalent_vector_with<cpo_result_t<round_t, T, rounding_t<R>>>;
    };

template <>
struct extended_impl<round_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, M>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <extended_vector T>
    requires unqualified_extended_cround<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return round(__DPL forward<T>(val));
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcround<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcround<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return round( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcround<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mcround<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return round(zero, cmask, __DPL forward<T>(val));
    }
    ///
    template <extended_vector T>
    requires unqualified_extended_croundne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding::no_exc_t) {
        return round(__DPL forward<T>(val), rounding::no_exc);
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcroundne<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), rounding::no_exc);
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcroundne<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return round( __DPL forward<S>(src), cmask, __DPL forward<T>(val),
            rounding::no_exc);
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcroundne<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            rounding::no_exc);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mcroundne<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return round(zero, cmask, __DPL forward<T>(val), rounding::no_exc);
    }

    ///
    template <extended_vector T, rounding_flags R>
    requires unqualified_extended_round<T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding_t<R> flags) {
        return round( __DPL forward<T>(val), flags);
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M, rounding_flags R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mround<S, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, rounding_t<R> flags) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), flags);
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T,
        rounding_flags R>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mround<S, cmask_t<S, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, cmask_t<S, M> cmask, T&& val, rounding_t<R> flags) {
        return round(
            __DPL forward<S>(src), cmask, __DPL forward<T>(val), flags);
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M, rounding_flags R>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mround<dx::zero_t, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, rounding_t<R> flags) {
        return round(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), flags);
    }

    template <extended_vector T, imask_t<T> M, rounding_flags R>
    requires unqualified_extended_mround<dx::zero_t, cmask_t<T, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val, rounding_t<R> flags) {
        return round(zero, cmask, __DPL forward<T>(val), flags);
    }
};

template <>
struct fallback_impl<round_t> {
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
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
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return operator()(val);
    }

    template <simd_abi A, simd_element_for<A> E, rounding_flags R>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<E, A> val, rounding_t<R>) {
        constexpr auto opt = rounding_v<R>;
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
DPL_EXPORT inline constexpr internal::round_t round{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

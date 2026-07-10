// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/copysign.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/rounding.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/math/details/constants.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES floor_t :
    private math_operation_base<floor_t>,
    private maskable_transform_base<floor_t> {
    using math_operation_base<floor_t>::operator();
    using maskable_transform_base<floor_t>::operator();
};

inline constexpr auto floor_noexc_v = rounding::to_neg_inf | rounding::no_exc;
inline constexpr auto floor_exc_v = rounding::to_neg_inf;

template <>
struct operation_signature<floor_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}

    static consteval void operator()(
        simd_vector auto&&, rounding::no_exc_t) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mfloor = cpo_invocable<floor_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<floor_t, T>>) && requires {
        {
            round(internal::abi<cpo_result_t<floor_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), floor_exc_v)
        } -> same_as<cpo_result_t<floor_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_canonical_mfloorne =
    cpo_invocable<floor_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        same_as<S, cpo_result_t<floor_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<floor_t, T, rounding::no_exc_t>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), floor_noexc_v)
        } -> same_as<cpo_result_t<floor_t, T, rounding::no_exc_t>>;
    };

template <>
struct canonical_impl<floor_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { round(internal::abi<A>, val, floor_exc_v); }
    {
        return round(internal::abi<A>, val, floor_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mfloor<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val, floor_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mfloor<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            floor_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mfloor<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, zero, mask, val, floor_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mfloor<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            floor_exc_v);
    }
    ///

    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept
    requires requires { round(internal::abi<A>, val, floor_noexc_v); }
    {
        return round(internal::abi<A>, val, floor_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mfloorne<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, mask_t<T> mask, T val,
        rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, floor_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mfloorne<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            floor_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mfloorne<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, mask, val, floor_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mfloorne<dx::zero_t, launder_cmask_t<T, M>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            floor_noexc_v);
    }
};

template <typename T>
concept unqualified_extended_floor = requires {
    {
        round(internal::declarg<T>(), floor_exc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept unqualified_extended_floorne = requires {
    {
        round(internal::declarg<T>(), floor_noexc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mfloor = cpo_invocable<floor_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<floor_t, T>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), floor_exc_v)
        } -> equivalent_vector_with<cpo_result_t<floor_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mfloorne =
    cpo_invocable<floor_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        equivalent_vector_with<S,
            cpo_result_t<floor_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), floor_noexc_v)
        }
        -> equivalent_vector_with<cpo_result_t<floor_t, T, rounding::no_exc_t>>;
    };

template <>
struct extended_impl<floor_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_floor<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return round(__DPL forward<T>(val), floor_exc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mfloor<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), floor_exc_v);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mfloor<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), floor_exc_v);
    }

    template <simd_vector T, result_mask_for<floor_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mfloor<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), floor_exc_v);
    }

    template <simd_vector T, result_cmask_for<floor_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mfloor<dx::zero_t,
            launder_cmask_t<cpo_result_t<floor_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return round(zero, dx::to_const_mask<cpo_result_t<floor_t, T>>(cmask),
            __DPL forward<T>(val), floor_exc_v);
    }

    template <extended_vector T>
    requires unqualified_extended_floorne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding::no_exc_t) {
        return round( __DPL forward<T>(val), floor_noexc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mfloorne<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), floor_noexc_v);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mfloorne<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), floor_noexc_v);
    }

    template <simd_vector T, result_mask_for<floor_t, T, rounding::no_exc_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mfloorne<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, rounding::no_exc_t) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            floor_noexc_v);
    }

    template <extended_vector T,
        result_cmask_for<floor_t, T, rounding::no_exc_t> M>
    requires extended_vector<T> &&
        unqualified_extended_mfloorne<dx::zero_t,
            launder_cmask_t<cpo_result_t<floor_t, T, rounding::no_exc_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, rounding::no_exc_t) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<floor_t, T, rounding::no_exc_t>>(
                cmask),
            __DPL forward<T>(val), floor_noexc_v);
    }
};

template <>
struct fallback_impl<floor_t> {
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<E, A> val) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto fr = finite - dx::trunc(finite);
        fr = dx::add(fr, fr < dx::zero, fr, dx::broadcast<E, A>(dx::one));

        auto const result = dx::copysign(finite - fr, finite);
        return dx::select(isfinite && dx::abs(val) < mx::maxint, result, val);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return operator()(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::floor_t floor{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

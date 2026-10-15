// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/compliance.h"
#include "dpl/core/math/details/floating_point_simd.h"
#include "dpl/core/math/details/ilogb.h"
#include "dpl/core/math/rounding.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwandnot.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct roundeven_t :
    public math_operation_base<roundeven_t>,
    public maskable_transform_base<roundeven_t> {
    using math_operation_base<roundeven_t>::operator();
    using maskable_transform_base<roundeven_t>::operator();
};

inline constexpr auto roundeven_noexc_v =
    rounding::to_nearest_int | rounding::no_exc;
inline constexpr auto roundeven_exc_v = rounding::to_nearest_int;

template <>
struct operation_signature<roundeven_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}

    static consteval void operator()(
        simd_vector auto&&, rounding::no_exc_t) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mroundeven = cpo_invocable<roundeven_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<roundeven_t, T>>) && requires {
        {
            round(internal::abi<cpo_result_t<roundeven_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), roundeven_exc_v)
        } -> same_as<cpo_result_t<roundeven_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_canonical_mroundevenne =
    cpo_invocable<roundeven_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        same_as<S, cpo_result_t<roundeven_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(
                internal::abi<cpo_result_t<roundeven_t, T, rounding::no_exc_t>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), roundeven_noexc_v)
        } -> same_as<cpo_result_t<roundeven_t, T, rounding::no_exc_t>>;
    };

template <>
struct canonical_impl<roundeven_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { round(internal::abi<T>, val, roundeven_exc_v); }
    {
        return round(internal::abi<T>, val, roundeven_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mroundeven<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val, roundeven_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mroundeven<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            roundeven_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mroundeven<dx::zero_t, simd_mask_type_t<T>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, zero, mask, val, roundeven_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mroundeven<dx::zero_t, launder_cmask_t<T, M>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            roundeven_exc_v);
    }
    ///

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, rounding::no_exc_t) noexcept
    requires requires { round(internal::abi<T>, val, roundeven_noexc_v); }
    {
        return round(internal::abi<T>, val, roundeven_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mroundevenne<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, roundeven_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mroundevenne<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            roundeven_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mroundevenne<dx::zero_t, simd_mask_type_t<T>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, mask, val, roundeven_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mroundevenne<dx::zero_t,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            roundeven_noexc_v);
    }
};

template <typename T>
concept unqualified_extended_roundeven = requires {
    {
        round(internal::declarg<T>(), roundeven_exc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept unqualified_extended_roundevenne = requires {
    {
        round(internal::declarg<T>(), roundeven_noexc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mroundeven = cpo_invocable<roundeven_t, T> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<roundeven_t, T>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), roundeven_exc_v)
        } -> equivalent_vector_with<cpo_result_t<roundeven_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mroundevenne =
    cpo_invocable<roundeven_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        equivalent_vector_with<S,
            cpo_result_t<roundeven_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), roundeven_noexc_v)
        } -> equivalent_vector_with<
            cpo_result_t<roundeven_t, T, rounding::no_exc_t>>;
    };

template <>
struct extended_impl<roundeven_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_roundeven<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return round(__DPL forward<T>(val), roundeven_exc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mroundeven<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), roundeven_exc_v);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mroundeven<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), roundeven_exc_v);
    }

    template <simd_vector T, result_mask_for<roundeven_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mroundeven<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            roundeven_exc_v);
    }

    template <simd_vector T, result_cmask_for<roundeven_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mroundeven<dx::zero_t,
            launder_cmask_t<cpo_result_t<roundeven_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<roundeven_t, T>>(cmask),
            __DPL forward<T>(val), roundeven_exc_v);
    }

    template <extended_vector T>
    requires unqualified_extended_roundevenne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding::no_exc_t) {
        return round( __DPL forward<T>(val), roundeven_noexc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mroundevenne<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), roundeven_noexc_v);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mroundevenne<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), roundeven_noexc_v);
    }

    template <simd_vector T,
        result_mask_for<roundeven_t, T, rounding::no_exc_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mroundevenne<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, rounding::no_exc_t) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            roundeven_noexc_v);
    }

    template <extended_vector T,
        result_cmask_for<roundeven_t, T, rounding::no_exc_t> M>
    requires extended_vector<T> &&
        unqualified_extended_mroundevenne<dx::zero_t,
            launder_cmask_t<cpo_result_t<roundeven_t, T, rounding::no_exc_t>,
                M>,
            T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, rounding::no_exc_t) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<roundeven_t, T, rounding::no_exc_t>>(
                cmask),
            __DPL forward<T>(val), roundeven_noexc_v);
    }
};

template <>
struct fallback_impl<roundeven_t> {
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        // round to nearest int, tie to nearest even
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto const floored = dx::floor(finite);

        // there are bit tricks alternatives to casting available but
        // they usually just add more instructions
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        auto const isodd = [](T floored) -> simd_mask_type_t<T> {
            auto const onei = dx::broadcast<mx::exponent_vector_t<T>>(dx::one);
            floored = dx::select(
                dx::cmplt(dx::abs(floored), mx::maxint), floored, dx::zero);
            return dx::cmpeq(
                dx::bwand(dx::element_cast<sint_t>(floored), onei), onei);
        }(floored);

        auto const fr = dx::subtract(finite, floored);
        auto const vhalf = dx::broadcast<T>(0.5);
        auto const roundup = dx::bwor(
            dx::cmpgt(fr, vhalf), dx::bwand(dx::cmpeq(fr, vhalf), isodd));

        auto const result =
            dx::add(floored, roundup, floored, dx::broadcast<T>(dx::one));
        return dx::copysign(val, isfinite, result, val);
    }

    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, rounding::no_exc_t) noexcept {
        return operator()(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::roundeven_t roundeven{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

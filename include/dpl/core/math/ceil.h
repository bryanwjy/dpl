// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/copysign.h"
#include "dpl/core/math/details/constants.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/rounding.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES ceil_t :
    public math_operation_base<ceil_t>,
    public maskable_transform_base<ceil_t> {
    using math_operation_base<ceil_t>::operator();
    using maskable_transform_base<ceil_t>::operator();
};

inline constexpr auto ceil_noexc_v = rounding::to_pos_inf | rounding::no_exc;
inline constexpr auto ceil_exc_v = rounding::to_pos_inf;

template <>
struct operation_signature<ceil_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}

    static consteval void operator()(
        simd_vector auto&&, rounding::no_exc_t) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mceil = cpo_invocable<ceil_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<ceil_t, T>>) && requires {
        {
            round(internal::abi<cpo_result_t<ceil_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), ceil_exc_v)
        } -> same_as<cpo_result_t<ceil_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_canonical_mceilne =
    cpo_invocable<ceil_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        same_as<S, cpo_result_t<ceil_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<ceil_t, T, rounding::no_exc_t>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), ceil_noexc_v)
        } -> same_as<cpo_result_t<ceil_t, T, rounding::no_exc_t>>;
    };

template <>
struct canonical_impl<ceil_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { round(internal::abi<T>, val, ceil_exc_v); }
    {
        return round(internal::abi<T>, val, ceil_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mceil<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val, ceil_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mceil<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            ceil_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mceil<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, zero, mask, val, ceil_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mceil<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            ceil_exc_v);
    }
    ///

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, rounding::no_exc_t) noexcept
    requires requires { round(internal::abi<T>, val, ceil_noexc_v); }
    {
        return round(internal::abi<T>, val, ceil_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mceilne<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, ceil_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mceilne<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            ceil_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mceilne<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, mask, val, ceil_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mceilne<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            ceil_noexc_v);
    }
};

template <typename T>
concept unqualified_extended_ceil = requires {
    {
        round(internal::declarg<T>(), ceil_exc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept unqualified_extended_ceilne = requires {
    {
        round(internal::declarg<T>(), ceil_noexc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mceil = cpo_invocable<ceil_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<ceil_t, T>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), ceil_exc_v)
        } -> equivalent_vector_with<cpo_result_t<ceil_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mceilne =
    cpo_invocable<ceil_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        equivalent_vector_with<S,
            cpo_result_t<ceil_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), ceil_noexc_v)
        }
        -> equivalent_vector_with<cpo_result_t<ceil_t, T, rounding::no_exc_t>>;
    };

template <>
struct extended_impl<ceil_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_ceil<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return round(__DPL forward<T>(val), ceil_exc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mceil<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), ceil_exc_v);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mceil<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), ceil_exc_v);
    }

    template <simd_vector T, result_mask_for<ceil_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mceil<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), ceil_exc_v);
    }

    template <simd_vector T, result_cmask_for<ceil_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mceil<dx::zero_t,
            launder_cmask_t<cpo_result_t<ceil_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return round(zero, dx::to_const_mask<cpo_result_t<ceil_t, T>>(cmask),
            __DPL forward<T>(val), ceil_exc_v);
    }

    template <extended_vector T>
    requires unqualified_extended_ceilne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding::no_exc_t) {
        return round( __DPL forward<T>(val), ceil_noexc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mceilne<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), ceil_noexc_v);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mceilne<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), ceil_noexc_v);
    }

    template <simd_vector T, result_mask_for<ceil_t, T, rounding::no_exc_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mceilne<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, rounding::no_exc_t) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            ceil_noexc_v);
    }

    template <extended_vector T,
        result_cmask_for<ceil_t, T, rounding::no_exc_t> M>
    requires unqualified_extended_mceilne<dx::zero_t,
        launder_cmask_t<cpo_result_t<ceil_t, T, rounding::no_exc_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, rounding::no_exc_t) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<ceil_t, T, rounding::no_exc_t>>(
                cmask),
            __DPL forward<T>(val), ceil_noexc_v);
    }
};

template <>
struct fallback_impl<ceil_t> {
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        auto const vone = dx::broadcast<T>(dx::one);
        auto fr = dx::subtract(val, dx::trunc(val));
        fr = dx::subtract(fr, dx::cmpgt(fr, dx::zero), fr, vone);

        auto const argvalid = dx::cmplt(dx::abs(val), mx::maxint);
        return dx::copysign(val, argvalid, dx::subtract(val, fr), val);
    }

    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, rounding::no_exc_t noexc) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto const vone = dx::broadcast<T>(dx::one);
        auto fr = dx::subtract(finite, dx::trunc(finite, noexc));
        fr = dx::subtract(fr, dx::cmpgt(fr, dx::zero), fr, vone);

        auto const argvalid = dx::cmplt(dx::abs(val), mx::maxint);
        return dx::copysign(val, argvalid, dx::subtract(finite, fr), finite);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::ceil_t ceil{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

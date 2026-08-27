// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/bitwise/bwandnot.h"
#include "dpl/core/operations/compare/max.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void abs(...) noexcept = delete;

struct DPL_EMPTY_BASES abs_t :
    public arithmetic_base<abs_t>,
    public maskable_transform_base<abs_t> {
    using operation_base<abs_t>::operator();
    using maskable_transform_base<abs_t>::operator();
};

template <>
struct operation_signature<abs_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<abs_t> {
    template <canonical_vector T>
    requires signed_integral<simd_element_type_t<T>> &&
        cpo_invocable<negate_t, T> && cpo_invocable<max_t, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        return dx::max(val, dx::negate(val));
    }

    template <canonical_vector T>
    requires unsigned_integral<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        return val;
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwandnot_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        using E = simd_element_type_t<T>;
        // signbit must be bit_representation_t<E>
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwandnot(val, signbit);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwandnot_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwandnot(src, mask, val, signbit);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwandnot_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwandnot(src, mask, val, signbit);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwandnot_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwandnot(zero, mask, val, signbit);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwandnot_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwandnot(zero, mask, val, signbit);
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mabs = cpo_invocable<abs_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<abs_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            abs(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<abs_t, T>>;
    };

template <>
struct canonical_impl<abs_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { abs(internal::abi<T>, val); }
    {
        return abs(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mabs<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return abs(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mabs<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return abs(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mabs<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return abs(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mabs<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return abs(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_abs = requires {
    { abs(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mabs = cpo_invocable<abs_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<abs_t, T>>) &&
    requires {
        {
            abs(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<abs_t, T>>;
    };

template <>
struct extended_impl<abs_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_abs<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return abs(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, vector_subsumed_by<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mabs<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return abs( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, vector_subsumed_by<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mabs<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return abs( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<abs_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mabs<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return abs(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<abs_t, T> M>
    requires unqualified_extended_mabs<dx::zero_t,
        launder_cmask_t<cpo_result_t<abs_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return abs(zero, dx::to_const_mask<cpo_result_t<abs_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::abs_t abs{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

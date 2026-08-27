// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/subtract.h"
#include "dpl/core/operations/bitwise/bwxor.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void negate(...) noexcept = delete;

struct DPL_EMPTY_BASES negate_t :
    public arithmetic_base<negate_t>,
    public maskable_transform_base<negate_t> {
    using operation_base<negate_t>::operator();
    using maskable_transform_base<negate_t>::operator();
};

template <>
struct operation_signature<negate_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct fallback_impl<negate_t> {
    template <canonical_vector T>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        return dx::subtract(dx::zero, val);
    }

    template <canonical_vector T>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return dx::subtract(src, mask, dx::zero, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        return dx::subtract(src, mask, dx::zero, val);
    }

    template <canonical_vector T>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return dx::subtract(zero, mask, zero, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, T val) noexcept {
        return dx::subtract(zero, mask, zero, val);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwxor_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        using E = simd_element_type_t<T>;
        // signbit must be bit_representation_t<E>
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(val, signbit);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwxor_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(src, mask, val, signbit);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwxor_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(src, mask, val, signbit);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwxor_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(zero, mask, val, signbit);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; } &&
        cpo_invocable<bwxor_t, T, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(zero, mask, val, signbit);
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mnegate = cpo_invocable<negate_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<negate_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            negate(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<negate_t, T>>;
    };

template <>
struct canonical_impl<negate_t> {
public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { negate(internal::abi<A>, val); }
    {
        return negate(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mnegate<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return negate(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mnegate<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return negate(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mnegate<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return negate(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mnegate<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return negate(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_negate = requires {
    { negate(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mnegate = cpo_invocable<negate_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<negate_t, T>>) &&
    requires {
        {
            negate(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<negate_t, T>>;
    };

template <>
struct extended_impl<negate_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_negate<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return negate(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, vector_subsumed_by<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mnegate<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return negate( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, vector_subsumed_by<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mnegate<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return negate( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<negate_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mnegate<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return negate(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<negate_t, T> M>
    requires unqualified_extended_mnegate<dx::zero_t,
        launder_cmask_t<cpo_result_t<negate_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return negate(zero, dx::to_const_mask<cpo_result_t<negate_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::negate_t negate{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

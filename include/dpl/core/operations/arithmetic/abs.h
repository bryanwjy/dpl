// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/compare/max.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
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
    template <fixed_width_abi A, simd_element_for<A> E>
    requires signed_integral<E> &&
        cpo_invocable<max_t, basic_vector<E, A>,
            cpo_result_t<negate_t, basic_vector<E, A>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        // Due to issues with nans, inf, signed zeros,
        // this is only allowed for ints
        return dx::max(val, dx::negate(val));
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires unsigned_integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> val) noexcept {
        return val;
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
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { abs(internal::abi<A>, val); }
    {
        return abs(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mabs<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return abs(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mabs<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return abs(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mabs<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
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

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mabs<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return abs( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
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
DPL_EXPORT inline constexpr internal::abs_t abs{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

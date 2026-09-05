// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/subtract.h"
#include "dpl/core/operations/bitwise/bwxor.h"
#include "dpl/core/operations/internal/broadcasting.h"
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
    requires cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        using E DPL_NODEBUG = simd_element_type_t<T>;
        if constexpr (floating_point_like<E> &&
            requires { floating_point_traits<E>::signbit; }) {
            constexpr auto signbit =
                __DPL bit_cast<E>(floating_point_traits<E>::signbit);
            return dx::bwxor(val, signbit);
        } else {
            return dx::subtract(dx::zero, val);
        }
    }

    template <canonical_vector T>
    requires cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        using E DPL_NODEBUG = simd_element_type_t<T>;
        if constexpr (floating_point_like<E> &&
            requires { floating_point_traits<E>::signbit; }) {
            constexpr auto signbit =
                __DPL bit_cast<E>(floating_point_traits<E>::signbit);
            return dx::bwxor(src, mask, val, signbit);
        } else {
            return dx::subtract(src, mask, dx::zero, val);
        }
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        using E DPL_NODEBUG = simd_element_type_t<T>;
        if constexpr (floating_point_like<E> &&
            requires { floating_point_traits<E>::signbit; }) {
            constexpr auto signbit =
                __DPL bit_cast<E>(floating_point_traits<E>::signbit);
            return dx::bwxor(src, mask, val, signbit);
        } else {
            return dx::subtract(src, mask, dx::zero, val);
        }
    }

    template <canonical_vector T>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        using E DPL_NODEBUG = simd_element_type_t<T>;
        if constexpr (floating_point_like<E> &&
            requires { floating_point_traits<E>::signbit; }) {
            constexpr auto signbit =
                __DPL bit_cast<E>(floating_point_traits<E>::signbit);
            return dx::bwxor(zero, mask, val, signbit);
        } else {
            return dx::subtract(zero, mask, zero, val);
        }
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires integral<simd_element_type_t<T>> &&
        cpo_invocable<subtract_t, dx::zero_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, T val) noexcept {
        using E DPL_NODEBUG = simd_element_type_t<T>;
        if constexpr (floating_point_like<E> &&
            requires { floating_point_traits<E>::signbit; }) {
            constexpr auto signbit =
                __DPL bit_cast<E>(floating_point_traits<E>::signbit);
            return dx::bwxor(zero, mask, val, signbit);
        } else {
            return dx::subtract(zero, mask, zero, val);
        }
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mnegate_base =
    cpo_invocable<negate_t, T> && requires {
        {
            negate(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<T>())
        } -> same_as<T>;
    };

template <typename M, typename T>
concept unqualified_canonical_mnegate =
    unqualified_canonical_mnegate_base<T, M, T>;

template <typename M, typename T>
concept unqualified_canonical_zmnegate =
    unqualified_canonical_mnegate_base<dx::zero_t, M, T>;

template <>
struct canonical_impl<negate_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { negate(internal::abi<T>, val); }
    {
        return negate(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mnegate<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return negate(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mnegate<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        return negate(internal::abi<T>, src, dx::to_const_mask<T>(mask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmnegate<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return negate(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmnegate<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M mask, T val) noexcept {
        return negate(internal::abi<T>, zero, dx::to_const_mask<T>(mask), val);
    }
};

template <typename T>
concept unqualified_extended_negate = requires {
    { negate(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mnegate_base =
    cpo_invocable<negate_t, T> && requires {
        {
            negate(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<T>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mnegate =
    equivalent_vector_with<T, S> && unqualified_extended_mnegate_base<S, M, T>;

template <typename M, typename T>
concept unqualified_extended_zmnegate =
    unqualified_extended_mnegate_base<dx::zero_t, M, T>;

template <>
struct extended_impl<negate_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_negate<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return negate(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mnegate<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return negate( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mnegate<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val) {
        return negate( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmnegate<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return negate(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zmnegate<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, T&& val) {
        return negate(zero, dx::to_const_mask<T>(mask), __DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::negate_t negate{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

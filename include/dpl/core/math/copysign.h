// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/signbit.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwxor.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void copysign(...) noexcept = delete;

struct DPL_EMPTY_BASES copysign_t :
    public math_operation_base<copysign_t>,
    public maskable_accumulation_base<copysign_t> {
    using math_operation_base<copysign_t>::operator();
    using maskable_accumulation_base<copysign_t>::operator();
};

template <>
struct operation_signature<copysign_t> {
    static consteval void operator()(
        simd_vector auto&&, simd_vector auto&&) noexcept {}
};

template <typename L, typename R>
concept unqualified_canonical_copysign = requires {
    {
        copysign(
            internal::abi<L>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<L>;
};

template <typename M, typename L, typename R>
concept unqualified_canonical_mcopysign =
    cpo_invocable<copysign_t, L, R> && requires {
        {
            copysign(internal::abi<L>, internal::declarg<L>(),
                internal::declarg<M>(), internal::declarg<R>())
        } -> same_as<L>;
    };

template <typename M, typename L, typename R>
concept unqualified_canonical_zmcopysign =
    cpo_invocable<copysign_t, L, R> && requires {
        {
            copysign(internal::abi<L>, dx::zero, internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<L>;
    };

template <>
struct canonical_impl<copysign_t> {
    template <canonical_vector L, vector_subsumed_by<L> R = L>
    requires canonical_vector<R> && unqualified_canonical_copysign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R rhs) noexcept {
        return copysign(internal::abi<L>, lhs, rhs);
    }

    template <canonical_vector L, vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_mcopysign<simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        L src, simd_mask_type_t<L> mask, R rhs) noexcept {
        return copysign(internal::abi<L>, src, mask, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M,
        vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_mcopysign<launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L src, M mask, R rhs) noexcept {
        return copysign(internal::abi<L>, src, dx::to_const_mask<L>(mask), rhs);
    }

    template <canonical_vector L, vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_zmcopysign<simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        dx::zero_t zero, simd_mask_type_t<L> mask, L lhs, R rhs) noexcept {
        return copysign(internal::abi<L>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M,
        vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_zmcopysign<launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        dx::zero_t zero, M mask, L lhs, R rhs) noexcept {
        return copysign(
            internal::abi<L>, zero, dx::to_const_mask<L>(mask), lhs, rhs);
    }
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_copysign = requires {
    {
        copysign(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename M, typename L, typename R>
concept unqualified_extended_mcopysign =
    cpo_invocable<copysign_t, L, R> && requires {
        {
            copysign(internal::declarg<L>(), internal::declarg<M>(),
                internal::declarg<R>())
        } -> vector_with_common_abi<simd_abi_type_t<L>>;
    };

template <typename M, typename L, typename R>
concept unqualified_extended_zmcopysign =
    cpo_invocable<copysign_t, L, R> && requires {
        {
            copysign(dx::zero, internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> vector_with_common_abi<simd_abi_type_t<L>>;
    };

template <>
struct extended_impl<copysign_t> {
public:
    template <simd_vector L, vector_subsumed_by<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_copysign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return copysign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M, vector_subsumed_by<L> R>
    requires (extended_vector<L> || extended_mask<M> || extended_vector<R>) &&
        unqualified_extended_mcopysign<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& src, M&& mask, R&& rhs) {
        return copysign(__DPL forward<L>(src), __DPL forward<M>(mask),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, const_mask_for<L> M, vector_subsumed_by<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcopysign<launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& src, M mask, R&& rhs) {
        return copysign(__DPL forward<L>(src), dx::to_const_mask<L>(mask),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M, vector_subsumed_by<L> R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmcopysign<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return copysign(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, const_mask_for<L> M, vector_subsumed_by<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmcopysign<launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return copysign(zero, dx::to_const_mask<L>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<copysign_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> lhs, T rhs) noexcept {
        if constexpr (unsigned_integral<simd_element_type_t<T>>) {
            return lhs;
        } else {
            return dx::negate(lhs, dx::signbit(dx::bwxor(lhs, rhs)), lhs);
        }
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> lhs, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(lhs, dx::bwand(dx::bwxor(lhs, rhs), signbit));
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(
            src, mask, src, dx::bwand(dx::bwxor(src, rhs), signbit));
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        type_identity_t<T> src, M mask, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(
            src, mask, src, dx::bwand(dx::bwxor(src, rhs), signbit));
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(dx::zero_t zero,
        simd_mask_type_t<T> mask, type_identity_t<T> lhs, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(
            dx::zero, mask, lhs, dx::bwand(dx::bwxor(lhs, rhs), signbit));
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires floating_point_like<simd_element_type_t<T>> &&
        requires { floating_point_traits<simd_element_type_t<T>>::signbit; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, type_identity_t<T> lhs, T rhs) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto signbit =
            __DPL bit_cast<E>(floating_point_traits<E>::signbit);
        return dx::bwxor(
            dx::zero, mask, lhs, dx::bwand(dx::bwxor(lhs, rhs), signbit));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::copysign_t copysign{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

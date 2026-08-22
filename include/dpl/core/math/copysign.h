// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/signbit.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwxor.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void copysign(...) noexcept = delete;

struct DPL_EMPTY_BASES copysign_t :
    public math_operation_base<copysign_t>,
    public maskable_transform_base<copysign_t> {
    using math_operation_base<copysign_t>::operator();
    using maskable_transform_base<copysign_t>::operator();
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

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_mcopysign = cpo_invocable<copysign_t, L, R> &&
    (!simd_type<S> || same_as<S, cpo_result_t<copysign_t, L, R>>) && requires {
        {
            copysign(internal::abi<cpo_result_t<copysign_t, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<copysign_t, L, R>>;
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
        unqualified_canonical_mcopysign<L, simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(type_identity_t<L> src,
        simd_mask_type_t<L> mask, L lhs, R rhs) noexcept {
        return copysign(internal::abi<L>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M,
        vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_mcopysign<L, launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        type_identity_t<L> src, M cmask, L lhs, R rhs) noexcept {
        return copysign(
            internal::abi<L>, src, dx::to_const_mask<L>(cmask), lhs, rhs);
    }

    template <canonical_vector L, vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_mcopysign<dx::zero_t, simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        dx::zero_t zero, simd_mask_type_t<L> mask, L lhs, R rhs) noexcept {
        return copysign(internal::abi<L>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M,
        vector_subsumed_by<L> R = L>
    requires canonical_vector<R> &&
        unqualified_canonical_mcopysign<dx::zero_t, launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return copysign(
            internal::abi<L>, zero, dx::to_const_mask<L>(cmask), lhs, rhs);
    }
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_copysign = requires {
    {
        copysign(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mcopysign = cpo_invocable<copysign_t, L, R> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<copysign_t, L, R>>) &&
    requires {
        {
            copysign(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<copysign_t, L, R>>;
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

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> L,
        vector_subsumed_by<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mcopysign<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return copysign(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> L,
        vector_subsumed_by<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcopysign<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return copysign(src, dx::to_const_mask<S>(cmask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, vector_subsumed_by<L> R,
        result_mask_for<copysign_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcopysign<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return copysign(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, vector_subsumed_by<L> R,
        result_cmask_for<copysign_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcopysign<dx::zero_t,
            launder_cmask_t<cpo_result_t<copysign_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return copysign(zero,
            dx::to_const_mask<cpo_result_t<copysign_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<copysign_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T lhs, type_identity_t<T> rhs) noexcept {
        if constexpr (unsigned_integral<simd_element_type_t<T>>) {
            return lhs;
        } else {
            return dx::negate(lhs, dx::signbit(dx::bwxor(lhs, rhs)), lhs);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::copysign_t copysign{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

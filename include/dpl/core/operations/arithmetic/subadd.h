// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/negate.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/const_mask.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Fused SIMD sub-add operation using a fixed lane-parity sign mask.
 *
 * Computes:
 *
 * CODE_BLOCK_BEGIN
 * result[i] = a[i] + ((i % 2 == 0) ? +b[i] : -b[i])
 * CODE_BLOCK_END
 *
 * This is the inverse lane-parity convention of @c subadd.
 *
 * @see subadd
 */
void subadd(...) noexcept = delete;

struct DPL_EMPTY_BASES subadd_t :
    public arithmetic_base<subadd_t>,
    public maskable_transform_base<subadd_t>,
    public binary_broadcastable_operation<subadd_t> {
    using operation_base<subadd_t>::operator();
    using maskable_transform_base<subadd_t>::operator();
    using binary_broadcastable_operation<subadd_t>::operator();
};

template <>
struct operation_signature<subadd_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<subadd_t> : binary_broadcasting_fallback<subadd_t> {
    using mask_t DPL_NODEBUG = cmask_t<0b0101>;
    static constexpr mask_t mask{};

    template <canonical_vector LT, canonical_vector RT>
    requires cpo_invocable<negate_t, RT, mask_t, RT> &&
        cpo_invocable<add_t, LT, cpo_result_t<negate_t, RT, mask_t, RT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(LT lhs, RT rhs) noexcept {
        return dx::add(lhs, dx::negate(rhs, mask, rhs));
    }

    using binary_broadcasting_fallback<subadd_t>::operator();
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_subadd = requires {
    {
        subadd(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> canonical_vector;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R>>
concept unqualified_canonical_msubadd = cpo_invocable<subadd_t, L, R> &&
    (!simd_type<S> || same_as<S, cpo_result_t<subadd_t, L, R>>) && requires {
        {
            subadd(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<cpo_result_t<subadd_t, L, R>>;
    };

template <>
struct canonical_impl<subadd_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_subadd<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_subadd<L, R, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return subadd(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_subadd<L, R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return subadd(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<result_t<L, R>, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<result_t<L, R>,
            launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<dx::zero_t,
            launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_subadd = requires {
    {
        subadd(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msubadd = cpo_invocable<subadd_t, L, R> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<subadd_t, L, R>>) &&
    requires {
        {
            subadd(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<subadd_t, L, R>>;
    };

template <>
struct extended_impl<subadd_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_subadd<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return subadd(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_subadd<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_subadd<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return subadd(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_msubadd<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return subadd(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <fixed_width_vector S, const_mask_for<S> M,
        common_vector_with<S> L, common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return subadd(src, dx::to_const_mask<S>(cmask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_mask_for<subadd_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return subadd(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_cmask_for<subadd_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<dx::zero_t,
            launder_cmask_t<cpo_result_t<subadd_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return subadd(zero,
            dx::to_const_mask<cpo_result_t<subadd_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::subadd_t subadd{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

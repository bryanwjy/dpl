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
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/cpo_result.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
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
    private arithmetic_base<subadd_t>,
    private maskable_transform_base<subadd_t>,
    private binary_broadcastable_operation<subadd_t> {
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
    static constexpr auto mask = imm<0b0101>;

    template <canonical_vector LT, canonical_vector RT>
    requires cpo_invocable<negate_t, RT, decltype(mask), RT> &&
        cpo_invocable<add_t, LT, cpo_result_t<negate_t, RT, decltype(mask), RT>>
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
concept unqualified_canonical_msubadd =
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<subadd_t, L, R>>) &&
    requires {
        {
            subadd(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<subadd_t, L, R>>;
    };

template <>
struct canonical_impl<subadd_t> {
private:
    template <typename L, typename R>
    using source_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> V>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, V>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_subadd<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_subadd<L, R, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R&& rhs) noexcept {
        return subadd(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_subadd<L, R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R rhs) noexcept {
        return subadd(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<source_t<L, R>, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        source_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<source_t<L, R>, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        source_t<L, R> src, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, src, cmask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_msubadd<dx::zero_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, zero, cmask, lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_subadd = requires {
    {
        subadd(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msubadd =
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
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<S>::size>;

    template <typename S, simask_t<S> M>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<S>::size, M>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = simask_t<cpo_result_t<subadd_t, L, R>>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = scmask_t<cpo_result_t<subadd_t, L, R>, M>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

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

    template <fixed_width_vector S, simask_t<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, scmask_t<S, M> cmask, L&& lhs, R&& rhs) {
        return subadd(
            src, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        common_mask_with<mask_t<L, R>> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return subadd(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<dx::zero_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return subadd(
            zero, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::subadd_t subadd{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

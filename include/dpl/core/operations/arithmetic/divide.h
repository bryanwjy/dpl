// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void divide(...) noexcept = delete;

struct DPL_EMPTY_BASES divide_t :
    private arithmetic_base<divide_t>,
    private maskable_transform_base<divide_t>,
    private binary_broadcastable_operation<divide_t> {
    using operation_base<divide_t>::operator();
    using maskable_transform_base<divide_t>::operator();
    using binary_broadcastable_operation<divide_t>::operator();
};

template <>
struct operation_signature<divide_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<divide_t> : binary_broadcasting_fallback<divide_t> {
    template <fixed_width_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](E lhs, E rhs) { return static_cast<E>(lhs / rhs); }, lhs, rhs);
    }

    using binary_broadcasting_fallback<divide_t>::operator();
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_divide = requires {
    {
        divide(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> canonical_vector;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R>>
concept unqualified_canonical_mdivide =
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<divide_t, L, R>>) &&
    requires {
        {
            divide(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<divide_t, L, R>>;
    };

template <>
struct canonical_impl<divide_t> {
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
    requires canonical_vector<R> && unqualified_canonical_divide<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        return divide(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_divide<L, R, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R&& rhs) noexcept {
        return divide(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_divide<L, R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R rhs) noexcept {
        return divide(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mdivide<source_t<L, R>, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        source_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return divide(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mdivide<source_t<L, R>, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        source_t<L, R> src, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return divide(internal::abi<common_abi_t<L, R>>, src, cmask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mdivide<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return divide(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mdivide<dx::zero_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return divide(internal::abi<common_abi_t<L, R>>, zero, cmask, lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_divide = requires {
    {
        divide(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mdivide =
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<divide_t, L, R>>) &&
    requires {
        {
            divide(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<divide_t, L, R>>;
    };

template <>
struct extended_impl<divide_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<S>::size>;

    template <typename S, simask_t<S> M>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<S>::size, M>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = simask_t<cpo_result_t<divide_t, L, R>>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = scmask_t<cpo_result_t<divide_t, L, R>, M>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_divide<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return divide(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_divide<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_divide<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return divide(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mdivide<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return divide(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <fixed_width_vector S, simask_t<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mdivide<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, scmask_t<S, M> cmask, L&& lhs, R&& rhs) {
        return divide(
            src, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        common_mask_with<mask_t<L, R>> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mdivide<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return divide(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mdivide<dx::zero_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return divide(
            zero, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::divide_t divide{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

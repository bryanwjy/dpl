// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/internal/extended_operations.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/operation_base.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void divide(...) noexcept = delete;

struct divide_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_divide = requires(L lhs, R rhs) {
    {
        divide(internal::abi<A>, lhs, rhs)
    } -> canonical_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_divide = requires(L lhs, R rhs) {
    { divide(lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename L, typename R>
concept expression_divide = (simd_expression<L> || simd_expression<R>) &&
    invocable<divide_t, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_divide =
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<divide_t, canonical_type_t<L>, canonical_type_t<R>>);

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_divide = unqualified_extended_divide<L, R, A> ||
    expression_divide<L, R> || decayable_divide<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mdivide = requires(S src, M mask, L lhs, R rhs) {
    {
        divide(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<divide_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mdivide = requires(S src, M mask, L lhs, R rhs) {
    { divide(src, mask, lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mdivide = (simd_expression<S> || simd_expression<M> ||
                                 simd_expression<L> || simd_expression<R>) &&
    invocable<divide_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mdivide =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<divide_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<divide_t,
        canonical_or_zero_t<S, operation_result_t<divide_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mdivide = unqualified_extended_mdivide<S, M, L, R, A> ||
    expression_mdivide<S, M, L, R> || decayable_mdivide<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<divide_t, L, R>>,
            operation_result_t<divide_t, L, R>>>
concept unqualified_canonical_imdivide = requires(S src, M mask, L lhs, R rhs) {
    {
        divide(internal::abi<A>, src,
            internal::to_const_mask<A, divide_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<divide_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<divide_t, L, R>>,
            operation_result_t<divide_t, L, R>>>
concept unqualified_extended_imdivide = requires(S src, M mask, L lhs, R rhs) {
    {
        divide(
            src, internal::to_const_mask<A, divide_t, S, L, R>(mask), lhs, rhs)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imdivide =
    (simd_expression<S> || simd_expression<L> || simd_expression<R>) &&
    invocable<divide_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<divide_t, L, R>>,
            operation_result_t<divide_t, L, R>>>
concept decayable_imdivide =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<divide_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<divide_t,
        canonical_or_zero_t<S, operation_result_t<divide_t, L, R>, A>, M,
        canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<divide_t, L, R>>,
            operation_result_t<divide_t, L, R>>>
concept extended_imdivide = unqualified_extended_imdivide<S, M, L, R, A> ||
    expression_imdivide<S, M, L, R> || decayable_imdivide<S, M, L, R, A>;

struct divide_t : private binary_operation_base<divide_t> {
private:
    friend binary_operation_base<divide_t>;

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || canonical_vector<L>) &&
        (!simd_class<R> || canonical_vector<R>) && requires(L lhs, R rhs) {
            { divide(internal::abi<A>, lhs, rhs) } -> vector_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return divide(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || extended_vector<L>) &&
        (!simd_class<R> || extended_vector<R>) &&
        unqualified_extended_divide<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept(
        noexcept(divide(lhs, rhs))) {
        return divide(lhs, rhs);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](E lhs, E rhs) { return static_cast<E>(lhs / rhs); }, lhs, rhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_divide<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return divide(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA> ||
            !arithmetic_type<E>) &&
        unqualified_canonical_divide<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return divide(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && extended_divide<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_divide<L, R>) {
            return divide(lhs, rhs);
        } else if constexpr (expression_divide<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<divide_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> lhs,
        basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mdivide<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<divide_t>(src, mask, lhs, rhs);
            } else {
                return divide(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<divide_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
                 scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        maskable_args<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_mdivide<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<ME, SA> mask, basic_vector<E, LA> lhs,
        basic_vector<E, RA> rhs) noexcept {
        return divide(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        maskable_args<S, M, L, R> && extended_mdivide<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mdivide<S, M, L, R>) {
            return divide(src, mask, lhs, rhs);
        } else if constexpr (expression_mdivide<S, M, L, R>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mdivide<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<divide_t>(mask, lhs, rhs);
            } else {
                return divide(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mdivide<zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return divide(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_mdivide<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mdivide<zero_t, M, L, R>) {
            return divide(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_mdivide<zero_t, M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<divide_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imdivide<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<divide_t>(src, mask, lhs, rhs);
            } else {
                return divide(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<divide_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi LA,
        common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_imdivide<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return divide(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_imdivide<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imdivide<S, M, L, R>) {
            return divide(src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imdivide<S, M, L, R>) {
            return operator()(
                dx::evaluate(src), mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imdivide<zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<divide_t>(mask, lhs, rhs);
            } else {
                return divide(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imdivide<zero_t, M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return divide(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<divide_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imdivide<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<divide_t, L, R>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imdivide<zero_t, M, L, R>) {
            return divide(dx::zero, cmask, lhs, rhs);
        } else if constexpr (expression_imdivide<zero_t, M, L, R>) {
            return operator()(cmask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                cmask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<divide_t, L, R>> M>
    requires invocable<divide_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::divide_t divide{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

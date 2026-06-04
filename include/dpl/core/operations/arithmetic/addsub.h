// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/extended_operations.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Performs a fused SIMD add-sub operation using a fixed lane-parity sign
 * mask.
 *
 * Computes a packed addition where the second operand is sign-modified on a
 * per-lane basis according to lane index parity.
 *
 * Formally, for lane index i:
 *
 * CODE_BLOCK_BEGIN
 * result[i] = a[i] + s[i] * b[i]
 * CODE_BLOCK_END
 *
 * where the sign pattern is fixed as:
 *
 * CODE_BLOCK_BEGIN
 * s[i] = (i % 2 == 0) ? -1 : +1
 * CODE_BLOCK_END
 *
 * This yields the following lane-wise behavior:
 *
 * CODE_BLOCK_BEGIN
 * [a0 - b0, a1 + b1, a2 - b2, a3 + b3, ...]
 * CODE_BLOCK_END
 *
 * This operation is NOT a sequence of alternating addition and subtraction
 * operations. It is a single SIMD addition with a compile-time sign mask
 * applied to the second operand.
 *
 * @note This definition is intentionally fixed and does not depend on operand
 * evaluation order.
 *
 * @note This convention differs from some SIMD ISA definitions where the
 * "addsub" family uses the opposite lane-parity sign pattern:
 *
 *       CODE_BLOCK_BEGIN
 *       [a0 + b0, a1 - b1, a2 + b2, a3 - b3, ...]
 *       CODE_BLOCK_END
 *
 *       In those ISAs, the opposite polarity is exposed either via a separate
 * instruction (e.g. subadd) or as a distinct encoding choice.
 *
 *       DPL standardizes a single fixed convention to avoid ambiguity in
 * cross-ABI behavior, and to align with reading-order lane parity (even = first
 * lane, odd = second lane).
 *
 * @param a First SIMD operand.
 * @param b Second SIMD operand.
 * @return SIMD value containing the fused add-sub result.
 *
 * @warning This operation is layout-sensitive: lane ordering directly
 * determines the sign mask.
 */
void addsub(...) noexcept = delete;

struct addsub_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_addsub = requires(L lhs, R rhs) {
    {
        addsub(internal::abi<A>, lhs, rhs)
    } -> canonical_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_addsub = requires(L lhs, R rhs) {
    { addsub(lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename L, typename R>
concept expression_addsub = (simd_expression<L> || simd_expression<R>) &&
    invocable<addsub_t, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_addsub =
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<addsub_t, canonical_type_t<L>, canonical_type_t<R>>);

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_addsub = unqualified_extended_addsub<L, R, A> ||
    expression_addsub<L, R> || decayable_addsub<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_maddsub = requires(S src, M mask, L lhs, R rhs) {
    {
        addsub(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_maddsub = requires(S src, M mask, L lhs, R rhs) {
    { addsub(src, mask, lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_maddsub = (simd_expression<S> || simd_expression<M> ||
                                 simd_expression<L> || simd_expression<R>) &&
    invocable<addsub_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_maddsub =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<addsub_t,
        canonical_or_zero_t<S, operation_result_t<addsub_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_maddsub = unqualified_extended_maddsub<S, M, L, R, A> ||
    expression_maddsub<S, M, L, R> || decayable_maddsub<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>>,
            operation_result_t<addsub_t, L, R>>>
concept unqualified_canonical_imaddsub = requires(S src, M mask, L lhs, R rhs) {
    {
        addsub(internal::abi<A>, src,
            internal::to_const_mask<A, addsub_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>>,
            operation_result_t<addsub_t, L, R>>>
concept unqualified_extended_imaddsub = requires(S src, M mask, L lhs, R rhs) {
    {
        addsub(
            src, internal::to_const_mask<A, addsub_t, S, L, R>(mask), lhs, rhs)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imaddsub =
    (simd_expression<S> || simd_expression<L> || simd_expression<R>) &&
    invocable<addsub_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>>,
            operation_result_t<addsub_t, L, R>>>
concept decayable_imaddsub =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<addsub_t,
        canonical_or_zero_t<S, operation_result_t<addsub_t, L, R>, A>, M,
        canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<addsub_t, L, R>>,
            operation_result_t<addsub_t, L, R>>>
concept extended_imaddsub = unqualified_extended_imaddsub<S, M, L, R, A> ||
    expression_imaddsub<S, M, L, R> || decayable_imaddsub<S, M, L, R, A>;

struct addsub_t : private binary_operation_base<addsub_t> {
private:
    friend binary_operation_base<addsub_t>;

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || canonical_vector<L>) &&
        (!simd_class<R> || canonical_vector<R>) && requires(L lhs, R rhs) {
            { addsub(internal::abi<A>, lhs, rhs) } -> vector_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return addsub(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || extended_vector<L>) &&
        (!simd_class<R> || extended_vector<R>) &&
        unqualified_extended_addsub<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept(
        noexcept(addsub(lhs, rhs))) {
        return addsub(lhs, rhs);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return dx::add(lhs, dx::negate(rhs, imm<0b1010>, rhs));
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_addsub<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return addsub(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA> ||
            !arithmetic_type<E>) &&
        unqualified_canonical_addsub<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return addsub(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && extended_addsub<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_addsub<L, R>) {
            return addsub(lhs, rhs);
        } else if constexpr (expression_addsub<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<addsub_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> lhs,
        basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_maddsub<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<addsub_t>(src, mask, lhs, rhs);
            } else {
                return addsub(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<addsub_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
                 scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        maskable_args<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_maddsub<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<ME, SA> mask, basic_vector<E, LA> lhs,
        basic_vector<E, RA> rhs) noexcept {
        return addsub(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        maskable_args<S, M, L, R> && extended_maddsub<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_maddsub<S, M, L, R>) {
            return addsub(src, mask, lhs, rhs);
        } else if constexpr (expression_maddsub<S, M, L, R>) {
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
        if constexpr (unqualified_canonical_maddsub<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<addsub_t>(mask, lhs, rhs);
            } else {
                return addsub(internal::abi<A>, dx::zero, mask, lhs, rhs);
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
        unqualified_canonical_maddsub<zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return addsub(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_maddsub<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_maddsub<zero_t, M, L, R>) {
            return addsub(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_maddsub<zero_t, M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<addsub_t, M, L, R>
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
        if constexpr (unqualified_canonical_imaddsub<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<addsub_t>(src, mask, lhs, rhs);
            } else {
                return addsub(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<addsub_t>(src, mask, lhs, rhs);
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
        unqualified_canonical_imaddsub<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return addsub(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_imaddsub<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imaddsub<S, M, L, R>) {
            return addsub(src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imaddsub<S, M, L, R>) {
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
        if constexpr (unqualified_canonical_imaddsub<zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<addsub_t>(mask, lhs, rhs);
            } else {
                return addsub(internal::abi<A>, dx::zero,
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
        unqualified_canonical_imaddsub<zero_t, M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return addsub(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<addsub_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imaddsub<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<addsub_t, L, R>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imaddsub<zero_t, M, L, R>) {
            return addsub(dx::zero, cmask, lhs, rhs);
        } else if constexpr (expression_imaddsub<zero_t, M, L, R>) {
            return operator()(cmask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                cmask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<addsub_t, L, R>> M>
    requires invocable<addsub_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::addsub_t addsub{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

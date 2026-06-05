// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/internal/operation_base.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void sign(...) noexcept = delete;

struct sign_t;

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_canonical_sign = requires(L lhs, R rhs) {
    {
        sign(internal::abi<A>, lhs, rhs)
    } -> canonical_arithmetic_result<L, L, A>;
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_sign = requires(L lhs, R rhs) {
    { sign(lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename L, typename R>
concept expression_sign = (simd_expression<L> || simd_expression<R>) &&
    invocable<sign_t, simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_sign =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<sign_t, canonical_type_t<L>, canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_sign = unqualified_extended_sign<L, R, A> ||
    expression_sign<L, R> || decayable_sign<L, R, A>;

/**
 * Returns the negation of the left argument if the right argument is
 * negative.
 *
 * If the element types are unsigned, the left argument will be returned
 * unmodified.
 *
 * For integral elements, if the right argument is 0 the result will also be
 * zero.
 *
 * For floating point elements, the result is unaffected by 0.0 but a
 * -0.0 value on the right will negate the left.
 */
struct sign_t :
    private binary_operation_base<sign_t>,
    private mx::masked_operation<sign_t> {
private:
    friend binary_operation_base<sign_t>;
    friend mx::masked_operation<sign_t>;

    template <simd_abi A, typename L, typename R>
    requires (canonical_vector<L> || canonical_vector<R>) &&
        requires(L lhs, R rhs) {
            { sign(internal::abi<A>, lhs, rhs) } -> vector_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return sign(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_sign<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept(
        noexcept(sign(lhs, rhs))) {
        return sign(lhs, rhs);
    }

    template <basic_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (floating_point<E>) {
            return lhs ^ (rhs & dx::msb);
        } else {
            auto const negated = dx::negate(lhs, rhs < dx::zero, lhs);
            return dx::select(rhs == dx::zero, dx::zero, negated);
        }
    }

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires mx::canonical_masked_math_operator<sign_t, S, M, L, R> &&
        requires(S src, M mask, L lhs, R rhs) {
            sign(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires mx::extended_masked_math_operator<sign_t, S, M, L, R> &&
        requires(S src, M mask, L lhs, R rhs) { sign(src, mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, L lhs, R rhs) noexcept {
        return sign(src, mask, lhs, rhs);
    }

    template <typename M, simd_vector L, simd_vector R>
    requires mx::canonical_masked_math_zoperator<sign_t, M, L, R> &&
        requires(M mask, L lhs, R rhs) {
            sign(internal::abi<common_abi_t<L, R>>, dx::zero, mask, lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, L lhs, R rhs) noexcept {
        return sign(
            internal::abi<common_abi_t<L, R>>, dx::zero, mask, lhs, rhs);
    }

    template <typename M, simd_vector L, simd_vector R>
    requires mx::extended_masked_math_zoperator<sign_t, M, L, R> &&
        requires(M mask, L lhs, R rhs) { sign(dx::zero, mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, L lhs, R rhs) noexcept {
        return sign(dx::zero, mask, lhs, rhs);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires basic_element<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_sign<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return sign(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, simd_element_for<LA> E, common_abi_with<LA> RA>
    requires simd_element_for<E, RA> &&
        (different_from<LA, RA> || !basic_element<E>) &&
        unqualified_canonical_sign<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return sign(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && extended_sign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_sign<L, R>) {
            return sign(lhs, rhs);
        } else if constexpr (expression_sign<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<sign_t>::operator();
    using mx::masked_operation<sign_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sign_t sign{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

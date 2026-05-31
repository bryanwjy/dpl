// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void sign(...) noexcept = delete;
template <typename A, typename L, typename R>
concept unqualified_sign =
    requires(L lhs, R rhs) { sign(internal::abi<A>, lhs, rhs); };

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
    requires requires(L lhs, R rhs) { sign(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return sign(internal::abi<A>, left, right);
    }

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(
            basic_vector<E, A> left, basic_vector<E, A> right) noexcept {
        if constexpr (floating_point<E>) {
            return left ^ (right & dx::msb);
        } else {
            auto const negated = dx::negate(left, right < dx::zero, left);
            return dx::select(right == dx::zero, dx::zero, negated);
        }
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::maskable_operator<sign_t, S, M, T> &&
        mx::canonical_operator_args<S, M, T> && requires(S src, M mask, T val) {
            sign(internal::abi<T>, src, mask, val);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sign(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::maskable_operator<sign_t, S, M, T> &&
        (!mx::canonical_operator_args<S, M, T>) &&
        requires(S src, M mask, T val) { sign(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sign(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::maskable_zoperator<sign_t, M, T> &&
        mx::canonical_zoperator_args<sign_t, M, T> &&
        requires(M mask, T val) { sign(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sign(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::maskable_zoperator<sign_t, M, T> &&
        (!mx::canonical_zoperator_args<sign_t, M, T>) &&
        requires(M mask, T val) { sign(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sign(dx::zero, mask, val);
    }

public:
    template <canonical_vector T>
    requires arithmetic_vector<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T left, T right) noexcept {
        if constexpr (unqualified_sign<T, T, T>) {
            if consteval {
                return fallback(left, right);
            } else {
                return sign(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires arithmetic_vector<L> && arithmetic_vector<R> &&
        common_order_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_sign<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept
        -> simd_with<typename L::value_type, common_abi_t<L, R>> auto {
        using A = common_abi_t<L, R>;
        return sign(internal::abi<A>, left, right);
    }

    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires arithmetic_vector<L> && arithmetic_vector<R> &&
        common_order_simd_with<L, R> &&
        (!canonical_vector<L> || !canonical_vector<R>) &&
        (!unqualified_sign<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept
        -> simd_with<typename L::value_type, common_abi_t<L, R>> auto {
        return operator()(dx::to_canonical(left), dx::to_canonical(right));
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

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/negate.h"
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
struct sign_t : binary_operation_base<sign_t> {
private:
    friend binary_operation_base<sign_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { sign(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return sign(internal::abi<A>, left, right);
    }

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        if constexpr (floating_point<E>) {
            return left ^ (right & dx::msb);
        } else {
            auto const negated =
                dx::select(right < dx::zero, dx::negate(left), left);
            return dx::bit_drop(right == dx::zero, negated);
        }
    }

public:
    template <basic_simd_type T>
    requires arithmetic_simd<T>
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

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires arithmetic_simd<L> && arithmetic_simd<R> &&
        common_order_simd_with<L, R> && only_unqualified<L, R> &&
        unqualified_sign<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept
        -> simd_with<typename L::value_type, common_abi_t<L, R>> auto {
        using A = common_abi_t<L, R>;
        return sign(internal::abi<A>, left, right);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires arithmetic_simd<L> && arithmetic_simd<R> &&
        common_order_simd_with<L, R> &&
        (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_sign<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept
        -> simd_with<typename L::value_type, common_abi_t<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
    }

    using binary_operation_base<sign_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sign_t sign{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

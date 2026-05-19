// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
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
struct addsub_t;

/**
 * @brief Fused SIMD sub-add operation using a fixed lane-parity sign mask.
 *
 * Computes:
 *
 * CODE_BLOCK_BEGIN
 * result[i] = a[i] + ((i % 2 == 0) ? +b[i] : -b[i])
 * CODE_BLOCK_END
 *
 * This is the inverse lane-parity convention of @c addsub.
 *
 * @see addsub
 */
struct subadd_t;
void addsub(...) noexcept = delete;
void subadd(...) noexcept = delete;

template <typename T, typename L, typename R>
concept addsub_result = simd_vector<T> &&
    requires {
        typename operation_result_t<add_t, L, R>;
        typename operation_result_t<subtract_t, L, R>;
    } &&
    same_as<typename T::value_type,
        typename operation_result_t<add_t, L, R>::value_type> &&
    same_as<typename T::value_type,
        typename operation_result_t<subtract_t, L, R>::value_type> &&
    same_as<typename T::abi_type,
        typename operation_result_t<add_t, L, R>::abi_type> &&
    same_as<typename T::abi_type,
        typename operation_result_t<subtract_t, L, R>::abi_type>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_addsub = requires(L lhs, R rhs) {
    { addsub(internal::abi<A>, lhs, rhs) } -> addsub_result<L, R>;
};
template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_subadd = requires(L lhs, R rhs) {
    { subadd(internal::abi<A>, lhs, rhs) } -> addsub_result<L, R>;
};

struct addsub_t : binary_operation_base<addsub_t> {
private:
    friend binary_operation_base<addsub_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_addsub<A, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return addsub(internal::abi<A>, left, right);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> left, basic_vector<E, A> right) noexcept {
        return dx::add(left, dx::negate(right, imm<0b1010>, right));
    }

public:
    template <floating_point_simd L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_addsub<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return addsub(internal::abi<A>, lhs, rhs);
                }
            } else {
                return addsub(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <floating_point_simd L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_addsub<L, R> ||
            unqualified_addsub<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_addsub<L, R>) {
            return addsub(internal::abi<A>, lhs, rhs);
        } else {
            return addsub(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<addsub_t>::operator();
};

struct subadd_t : binary_operation_base<subadd_t> {
private:
    friend binary_operation_base<subadd_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_subadd<A, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return subadd(internal::abi<A>, left, right);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> left, basic_vector<E, A> right) noexcept {
        return dx::add(left, dx::negate(right, imm<0b0101>, right));
    }

    template <simd_vector L, simd_vector R>
    using result_for DPL_NODEBUG =
        common_arithmetic_simd_t<canonical_type_t<L>, canonical_type_t<R>>;

public:
    template <floating_point_simd L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_subadd<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return subadd(internal::abi<A>, lhs, rhs);
                }
            } else {
                return subadd(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <floating_point_simd L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_subadd<L, R> ||
            unqualified_subadd<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_subadd<L, R>) {
            return subadd(internal::abi<A>, lhs, rhs);
        } else {
            return subadd(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<subadd_t>::operator();
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END

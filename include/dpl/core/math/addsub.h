// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/negate.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void addsub(...) noexcept = delete;
void subadd(...) noexcept = delete;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_addsub = requires(L lhs, R rhs) {
    { addsub(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};
template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_subadd = requires(L lhs, R rhs) {
    { subadd(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
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
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        return dx::add(left, dx::negatei<0b0101>(right));
    }

public:
    template <floating_point_simd L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_addsub<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return addsub(internal::abi<A>, lhs, rhs);
                }
            } else {
                return addsub(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <floating_point_simd L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_addsub<L, R> ||
            unqualified_addsub<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_addsub<L, R>) {
            return addsub(internal::abi<A>, lhs, rhs);
        } else {
            return addsub(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
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
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        return dx::add(left, dx::negatei<0b1010>(right));
    }

    template <simd_type L, simd_type R>
    using result_for DPL_NODEBUG =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <floating_point_simd L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_subadd<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return subadd(internal::abi<A>, lhs, rhs);
                }
            } else {
                return subadd(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <floating_point_simd L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_subadd<L, R> ||
            unqualified_subadd<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_subadd<L, R>) {
            return subadd(internal::abi<A>, lhs, rhs);
        } else {
            return subadd(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<subadd_t>::operator();
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END

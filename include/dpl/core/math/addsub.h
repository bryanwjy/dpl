// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
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

struct addsub_t : binary_operation_base<addsub_t> {
private:
    friend binary_operation_base<addsub_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { addsub(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return addsub(internal::abi<A>, left, right);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        return dx::add(left, dx::negatei<0b0101>(right));
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T left, T right) noexcept {
        if constexpr (requires { addsub(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return addsub(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    requires floating_point_simd<L> && floating_point_simd<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return addsub(internal::abi<A>, left, right);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires floating_point_simd<L> && floating_point_simd<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { addsub(internal::abi<A>, left, right); }) {
            return addsub(internal::abi<A>, left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    using binary_operation_base<addsub_t>::operator();
};

struct subadd_t : binary_operation_base<subadd_t> {
private:
    friend binary_operation_base<subadd_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { subadd(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return subadd(internal::abi<A>, left, right);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        return dx::add(left, dx::negatei<0b1010>(right));
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T left, T right) noexcept {
        if constexpr (requires { subadd(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return subadd(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    requires floating_point_simd<L> && floating_point_simd<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return subadd(internal::abi<A>, left, right);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires floating_point_simd<L> && floating_point_simd<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { subadd(internal::abi<A>, left, right); }) {
            return subadd(internal::abi<A>, left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    using binary_operation_base<subadd_t>::operator();
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END

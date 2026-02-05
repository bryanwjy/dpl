// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_order_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void min(...) noexcept = delete;
void max(...) noexcept = delete;

template <typename A, typename L, typename R>
concept unqualified_min =
    requires(L lhs, R rhs) { min(internal::abi<A>, lhs, rhs); };

template <typename A, typename L, typename R>
concept unqualified_max =
    requires(L lhs, R rhs) { max(internal::abi<A>, lhs, rhs); };

struct min_t : binary_operation_base<min_t> {
private:
    friend binary_operation_base<min_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_min<A, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return min(internal::abi<A>, left, right);
    }

    template <simd_element L, simd_element R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            left, right, [](auto lhs, auto rhs) {
                return static_cast<T>(lhs < rhs ? lhs : rhs);
            });
    }

    template <simd_type L, simd_type R>
    using result_for = common_order_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type L, common_order_simd_with<L> R>
    requires basic_simd_type<R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_order_simd_t<L, R>;
        if constexpr (requires { min(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return min(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_type L, common_order_simd_with<L> R>
    requires only_unqualified<L, R> && unqualified_min<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return min(internal::abi<A>, left, right);
    }

    template <simd_type L, common_order_simd_with<L> R>
    requires (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_min<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
    }

    using binary_operation_base<min_t>::operator();
};

struct max_t : binary_operation_base<max_t> {
private:
    friend binary_operation_base<max_t>;

    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { max(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return max(internal::abi<A>, left, right);
    }

    template <simd_element L, simd_element R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_order_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            left, right, [](auto lhs, auto rhs) {
                return static_cast<T>(lhs > rhs ? lhs : rhs);
            });
    }

    template <simd_type L, simd_type R>
    using result_for = common_order_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type L, common_order_simd_with<L> R>
    requires basic_simd_type<R> && same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_order_simd_t<L, R>;
        if constexpr (requires { max(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return max(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <simd_type L, common_order_simd_with<L> R>
    requires only_unqualified<L, R> && unqualified_max<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return max(internal::abi<A>, left, right);
    }

    template <simd_type L, common_order_simd_with<L> R>
    requires (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_max<common_abi_t<L, R>, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        return operator()(dx::to_basic_type(left), dx::to_basic_type(right));
    }

    using binary_operation_base<max_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::min_t min{};
DPL_EXPORT inline constexpr internal::max_t max{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

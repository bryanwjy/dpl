// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/select.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
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

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_min = requires(L lhs, R rhs) {
    {
        min(internal::abi<A>, lhs, rhs)
    } -> equivalent_simd_as<common_order_simd_t<L, R>>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_max = requires(L lhs, R rhs) {
    {
        max(internal::abi<A>, lhs, rhs)
    } -> equivalent_simd_as<common_order_simd_t<L, R>>;
};

struct min_t : binary_operation_base<min_t> {
private:
    friend binary_operation_base<min_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_min<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return min(internal::abi<A>, left, right);
    }

    template <simd_element L, simd_element R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return dx::reinterpret<T>(dx::select(lhs < rhs, lhs, rhs));
    }

    template <simd_type L, simd_type R>
    using result_for = common_order_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <simd_type L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept {
        using T = common_order_simd_t<L, R>;
        using A = typename T::abi_type;
        if constexpr (unqualified_min<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(min(
                        internal::abi<T>, left, right))::value_type;
                    return dx::reinterpret<E>(fallback(left, right));
                } else {
                    return min(internal::abi<T>, left, right);
                }
            } else {
                return min(internal::abi<T>, left, right);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    template <simd_type L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_min<L, R> ||
            unqualified_min<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_min<L, R>) {
            return min(internal::abi<A>, lhs, rhs);
        } else {
            return min(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_order_type_t<L, R>;
        return dx::reinterpret<T>(dx::select(rhs < lhs, lhs, rhs));
    }

    template <simd_type L, simd_type R>
    using result_for = common_order_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <simd_type L, common_order_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_simd<L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L left, R right) noexcept {
        using T = common_order_simd_t<L, R>;
        using A = typename T::abi_type;
        if constexpr (unqualified_max<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(max(
                        internal::abi<T>, left, right))::value_type;
                    return dx::reinterpret<E>(fallback(left, right));
                } else {
                    return max(internal::abi<T>, left, right);
                }
            } else {
                return max(internal::abi<T>, left, right);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    template <simd_type L, common_order_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_max<L, R> ||
            unqualified_max<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_max<L, R>) {
            return max(internal::abi<A>, lhs, rhs);
        } else {
            return max(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
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

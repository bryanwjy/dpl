// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/hsum.h"
#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void inner_product(...) noexcept = delete;
template <auto>
void inner_product(...) noexcept = delete;

template <typename A, typename L, typename R>
concept unqualified_inner_product =
    requires(L lhs, R rhs) { inner_product(internal::abi<A>, lhs, rhs); };

template <typename M, typename A, typename L, typename R>
concept unqualified_inner_producti = immediate_mask_for<M, L> &&
    immediate_mask_for<M, R> && requires(L lhs, R rhs) {
        inner_product<immediate_mask_v<M, L>>(internal::abi<A>, lhs, rhs);
    };

template <typename L, typename R, typename... Args>
concept basic_inner_product =
    simd_type<L> && (... && immediate_mask_for<Args, L>) &&
    (... && immediate_mask_for<Args, R>) &&
    requires(basic_type_t<L> lhs, basic_type_t<L> rhs, Args... mask) {
        dx::reduce(mask..., dx::multiply(lhs, rhs), dx::add);
    };

struct inner_product_t {
    template <basic_simd_type T>
    requires basic_inner_product<T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T lhs, T rhs) noexcept {
        if constexpr (unqualified_inner_product<T, T, T>) {
            if consteval {
                return static_cast<T>(dx::hsum(dx::multiply(lhs, rhs)));
            } else {
                return static_cast<T>(
                    inner_product(internal::abi<T>, lhs, rhs));
            }
        } else {
            return dx::hsum(dx::multiply(lhs, rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires only_unqualified<L, R> &&
        unqualified_inner_product<common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L lhs, R rhs) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        using A = common_abi_t<L, R>;
        return inner_product(internal::abi<A>, lhs, rhs);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_inner_product<common_abi_t<L, R>, L, R>) &&
        basic_inner_product<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
    }

    template <basic_simd_type T, immediate_mask_for<T> M>
    requires basic_inner_product<T, T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T lhs, T rhs) noexcept {
        if constexpr (unqualified_inner_producti<M, T, T, T>) {
            if consteval {
                return static_cast<T>(dx::hsum(mask, dx::multiply(lhs, rhs)));
            } else {
                constexpr auto V = dx::immediate_mask_v<T, M>;
                return static_cast<T>(
                    inner_product<V>(internal::abi<T>, lhs, rhs));
            }
        } else {
            return dx::hsum(mask, dx::multiply(lhs, rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R,
        immediate_mask_for<L> M>
    requires immediate_mask_for<R, M> && only_unqualified<L, R> &&
        unqualified_inner_producti<M, common_abi_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, L lhs, R rhs) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        using A = common_abi_t<L, R>;
        constexpr auto V = dx::immediate_mask_v<L, M>;
        return inner_product<V>(internal::abi<A>, lhs, rhs);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R,
        immediate_mask_for<L> M>
    requires immediate_mask_for<R, M> &&
        (!basic_simd_type<L> || !basic_simd_type<R>) &&
        (!unqualified_inner_producti<M, common_abi_t<L, R>, L, R>) &&
        basic_inner_product<L, R, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        return operator()(mask, dx::to_basic_type(lhs), dx::to_basic_type(rhs));
    }
};

template <auto V>
struct inner_producti_t {};
template <integral auto V>
struct inner_producti_t<V> : binary_operation_base<inner_producti_t<V>> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires requires {
        typename mask_type<L>;
        requires regular_invocable<inner_product_t, mask_type<L>, L, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        constexpr mask_type<L> mask{};
        return inner_product_t::operator()(mask, lhs, rhs);
    }

    using binary_operation_base<inner_producti_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::inner_product_t inner_product{};
DPL_EXPORT template <auto V>
inline constexpr internal::inner_producti_t<V> inner_producti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

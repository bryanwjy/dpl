// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/hsum.h"
#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void inner_product(...) noexcept = delete;
template <auto>
void inner_product(...) noexcept = delete;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_inner_product = requires(L lhs, R rhs) {
    { inner_product(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_inner_producti = immediate_mask_for<M, L> &&
    immediate_mask_for<M, R> && requires(L lhs, R rhs) {
        {
            inner_product<immediate_mask_v<M, L>>(internal::abi<A>, lhs, rhs)
        } -> arithmetic_result<L, R>;
    };

template <typename L, typename R, typename... Args>
concept basic_inner_product =
    simd_type<L> && (... && immediate_mask_for<Args, L>) &&
    (... && immediate_mask_for<Args, R>) &&
    requires(basic_type_t<L> lhs, basic_type_t<L> rhs, Args... mask) {
        dx::reduce(mask..., dx::multiply(lhs, rhs), dx::add);
    };

struct inner_product_t {
private:
    template <simd_element L, simd_element R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<L, A> rhs) noexcept {
        return dx::hsum(dx::multiply(lhs, rhs));
    }

    template <auto V, simd_element L, simd_element R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<L, A> lhs, basic_simd<L, A> rhs) noexcept {
        return dx::hsumi<V>(dx::multiply(lhs, rhs));
    }

public:
    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_inner_product<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using RT =
                        decltype(inner_product(internal::abi<A>, lhs, rhs));
                    return dx::reinterpret<RT>(fallback(lhs, rhs));
                } else {
                    return inner_product(internal::abi<A>, lhs, rhs);
                }
            } else {
                return inner_product(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_inner_product<L, R> ||
            unqualified_inner_product<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_inner_product<L, R>) {
            return inner_product(internal::abi<A>, lhs, rhs);
        } else {
            return inner_product(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R,
        immediate_mask_for<L> M>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        constexpr auto V = immediate_mask_v<L, M>;
        if constexpr (unqualified_inner_producti<M, L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using RT =
                        decltype(inner_product<V>(internal::abi<A>, lhs, rhs));
                    return dx::reinterpret<RT>(fallbacki<V>(lhs, rhs));
                } else {
                    return inner_product<V>(internal::abi<A>, lhs, rhs);
                }
            } else {
                return inner_product<V>(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallbacki<V>(lhs, rhs);
        } else {
            return operator()(
                mask, dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R,
        immediate_mask_for<L> M>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_inner_producti<M, L, R> ||
            unqualified_inner_producti<M, basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M, L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        constexpr auto V = immediate_mask_v<L, M>;
        if constexpr (unqualified_inner_producti<M, L, R>) {
            return inner_product<V>(internal::abi<A>, lhs, rhs);
        } else {
            return inner_product<V>(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
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
    template <fixed_width_simd L, common_arithmetic_simd_with<L> R>
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

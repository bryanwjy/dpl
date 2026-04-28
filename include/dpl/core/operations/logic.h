// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/logic_reduction.h" // IWYU pragma: export
#include "dpl/core/operations/negated_mask.h"    // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/to_native_type.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_mask_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T, typename L, typename R>
concept logical_result = common_size_simd_with<T, common_size_simd_t<L, R>> &&
    same_abi_as<common_abi_t<L, R>, typename T::abi_type>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_logical_or = requires(L lhs, R rhs) {
    { logical_or(internal::abi<A>, lhs, rhs) } -> logical_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_logical_and = requires(L lhs, R rhs) {
    { logical_and(internal::abi<A>, lhs, rhs) } -> logical_result<L, R>;
};

template <typename T>
concept unqualified_logical_not = requires(T val) {
    { logical_not(internal::abi<T>, val) } -> common_size_simd_with<T>;
};

struct logical_and_t {
    template <simd_mask_type L, simd_mask_type R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_logical_and<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = simd_element_type_t<decltype(logical_and(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(dx::bwand(lhs, rhs));
                } else {
                    return logical_and(internal::abi<A>, lhs, rhs);
                }
            } else {
                return logical_and(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return dx::bwand(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, simd_mask_type R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_logical_and<L, R> ||
            unqualified_logical_and<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_logical_and<L, R, A>) {
            return logical_and(internal::abi<A>, lhs, rhs);
        } else {
            return logical_and(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }
};

struct logical_or_t {
    template <simd_mask_type L, simd_mask_type R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_logical_or<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = simd_element_type_t<decltype(logical_or(
                        internal::abi<A>, lhs, rhs))>;
                    return dx::reinterpret<E>(dx::bwor(lhs, rhs));
                } else {
                    return logical_or(internal::abi<A>, lhs, rhs);
                }
            } else {
                return logical_or(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return dx::bwor(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_mask_type L, simd_mask_type R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_logical_or<L, R> ||
            unqualified_logical_or<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_logical_or<L, R, A>) {
            return logical_or(internal::abi<A>, lhs, rhs);
        } else {
            return logical_or(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }
};

struct logical_not_t {
    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_logical_not<T>) {
            if constexpr (basic_simd_mask_type<T>) {
                if consteval {
                    using E = simd_element_type_t<decltype(logical_not(
                        internal::abi<T>, val))>;
                    return dx::reinterpret<E>(dx::bwnot(val));
                } else {
                    return logical_not(internal::abi<T>, val);
                }
            } else {
                return logical_not(internal::abi<T>, val);
            }
        } else {
            return negated_mask<M>(val);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::logical_and_t logical_and{};
DPL_EXPORT inline constexpr internal::logical_or_t logical_or{};
DPL_EXPORT inline constexpr internal::logical_not_t logical_not{};
} // namespace cpo

DPL_EXPORT template <typename D>
class logical_simd_interface {
public:
    template <simd_mask_type R>
    requires regular_invocable<internal::logical_or_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_or_t, D, R> operator||(
        this D lhs, R rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_or(lhs, rhs);
    }

    template <simd_mask_type R>
    requires regular_invocable<internal::logical_and_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_and_t, D, R> operator&&(
        this D lhs, R rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_and(lhs, rhs);
    }

    template <simd_mask_type Self>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_not_t, Self> operator!(
        this Self lhs) noexcept {
        return datapar::logical_not(lhs);
    }

    template <simd_mask_type L>
    requires regular_invocable<internal::logical_or_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::logical_or_t, L, D> operator||(
        L lhs, D rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_or(lhs, rhs);
    }

    template <simd_mask_type L>
    requires regular_invocable<internal::logical_or_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::logical_and_t, L, D> operator&&(
        L lhs, D rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_and(lhs, rhs);
    }
};

DPL_EXPORT template <simd_mask_type L, simd_mask_type R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwor_t, L, R> operator||(
    L lhs, R rhs) noexcept {
    return datapar::logical_or(lhs, rhs);
}

DPL_EXPORT template <simd_mask_type L, simd_mask_type R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwand_t, L, R> operator&&(
    L lhs, R rhs) noexcept {
    return datapar::logical_and(lhs, rhs);
}

DPL_EXPORT template <simd_mask_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator!(T val) noexcept
    -> invoke_result_t<internal::logical_not_t, T> {
    return datapar::logical_not(val);
}

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

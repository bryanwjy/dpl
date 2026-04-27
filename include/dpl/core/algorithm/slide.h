// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/common_order_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t>
void slide_left(...) noexcept = delete;
void slide_left(...) noexcept = delete;
template <size_t>
void slide_right(...) noexcept = delete;
void slide_right(...) noexcept = delete;

template <typename T, typename L, typename R>
concept slide_result = simd_with<R, typename L::value_type> &&
    simd_with<T, typename L::value_type, common_abi_t<L, R>>;

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_lefti = requires(L lhs, R rhs) {
    { slide_left<V::value>(internal::abi<A>, lhs, rhs) } -> slide_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_left = requires(L lhs, R rhs, size_t count) {
    { slide_left(internal::abi<A>, lhs, rhs, count) } -> slide_result<L, R>;
};

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_righti = requires(L lhs, R rhs) {
    { slide_right<V::value>(internal::abi<A>, lhs, rhs) } -> slide_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_right = requires(L lhs, R rhs, size_t count) {
    { slide_right(internal::abi<A>, lhs, rhs, count) } -> slide_result<L, R>;
};

struct slide_left_t {
private:
    template <simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd<E, A> lhs, basic_simd<E, A> rhs, size_t num) noexcept {
        num = num <= element_count<E, A> ? num : element_count<E, A>;
        auto const low = dx::shift_left(lhs, num);
        auto const high = dx::shift_right(rhs, element_count<E, A> - num);
        return dx::reinterpret<E>(dx::bwor(low, high));
    }

    template <size_t N, simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> lhs, basic_simd<E, A> rhs) noexcept {
        static_assert(N <= element_count<E, A>);
        constexpr auto size = 2 * element_count<E, A>;
        auto const low = dx::shift_left(lhs, imm<N>);
        auto const high = dx::shift_right(rhs, imm<element_count<E, A> - N>);
        return dx::reinterpret<E>(dx::bwor(low, high));
    }

public:
    template <simd_type L, simd_with<typename L::value_type> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_left<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(slide_left(
                        internal::abi<A>, lhs, rhs, num))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return slide_left(internal::abi<A>, lhs, rhs, num);
                }
            } else {
                return slide_left(internal::abi<A>, lhs, rhs, num);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs, num);
        } else {
            return operator()(
                dx::to_basic_type(lhs), dx::to_basic_type(rhs), num);
        }
    }

    template <simd_type L, simd_with<typename L::value_type> R>
    requires (!same_abi_simd_as<L, R> || scalable_simd<common_abi_t<L, R>>) &&
        (unqualified_slide_left<L, R> ||
            unqualified_slide_left<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_left<L, R>) {
            return slide_left(internal::abi<A>, lhs, rhs, num);
        } else {
            return slide_left(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs), num);
        }
    }

    template <simd_type L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_lefti<N, L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(slide_left<N::value>(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback<N::value>(lhs, rhs));
                } else {
                    return slide_left<N::value>(internal::abi<A>, lhs, rhs);
                }
            } else {
                return slide_left<N::value>(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback<N::value>(lhs, rhs);
        } else {
            return operator()(
                dx::to_basic_type(lhs), dx::to_basic_type(rhs), num);
        }
    }

    template <simd_type L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires (!same_abi_simd_as<L, R> || scalable_simd<common_abi_t<L, R>>) &&
        (unqualified_slide_lefti<N, L, R> ||
            unqualified_slide_lefti<N, basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_left<L, N>) {
            return slide_left<N::value>(internal::abi<A>, lhs, rhs);
        } else {
            return slide_left<N::value>(internal::abi<A>,
                dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }
};

struct slide_right_t {
private:
    template <simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_simd<E, A> lhs, basic_simd<E, A> rhs, size_t num) noexcept {
        num = num <= element_count<E, A> ? num : element_count<E, A>;
        return slide_left_t::operator()(lhs, rhs, element_count<E, A> - num);
    }

    template <size_t N, simd_element E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<E, A> lhs, basic_simd<E, A> rhs) noexcept {
        constexpr auto num = element_count<E, A> - N;
        return slide_left_t::operator()(lhs, rhs, imm<num>);
    }

public:
    template <simd_type L, simd_with<typename L::value_type> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_right<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(slide_right(
                        internal::abi<A>, lhs, rhs, num))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return slide_right(internal::abi<A>, lhs, rhs, num);
                }
            } else {
                return slide_right(internal::abi<A>, lhs, rhs, num);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs, num);
        } else {
            return operator()(
                dx::to_basic_type(lhs), dx::to_basic_type(rhs), num);
        }
    }

    template <simd_type L, simd_with<typename L::value_type> R>
    requires (!same_abi_simd_as<L, R> || scalable_simd<common_abi_t<L, R>>) &&
        (unqualified_slide_right<L, R> ||
            unqualified_slide_right<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_right<L, R>) {
            return slide_right(internal::abi<A>, lhs, rhs, num);
        } else {
            return slide_right(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs), num);
        }
    }

    template <simd_type L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_righti<N, L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    using E = typename decltype(slide_right<N::value>(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallbacki<N::value>(lhs, rhs));
                } else {
                    return slide_right<N::value>(internal::abi<A>, lhs, rhs);
                }
            } else {
                return slide_right<N::value>(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallbacki<N::value>(lhs, rhs);
        } else {
            return operator()(
                dx::to_basic_type(lhs), dx::to_basic_type(rhs), num);
        }
    }

    template <simd_type L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires (!same_abi_simd_as<L, R> || scalable_simd<common_abi_t<L, R>>) &&
        (unqualified_slide_righti<N, L, R> ||
            unqualified_slide_righti<N, basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_right<L, N>) {
            return slide_right<N::value>(internal::abi<A>, lhs, rhs);
        } else {
            return slide_right<N::value>(internal::abi<A>,
                dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }
};

template <size_t V>
struct slide_lefti_t : binary_operation_base<slide_lefti_t<V>> {
private:
    friend binary_operation_base<slide_lefti_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept
    requires requires {
        { slide_left<V>(internal::abi<A>, lhs, rhs) } -> simd_with_abi<A>;
    }
    {
        return slide_left<V>(internal::abi<A>, lhs, rhs);
    }

public:
    template <simd_type L, simd_with<typename L::value_type> R>
    requires regular_invocable<slide_left_t, L, R, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        return slide_left_t::operator()(lhs, rhs, imm<V>);
    }

    using binary_operation_base<slide_lefti_t>::operator();
};

template <size_t V>
struct slide_righti_t : binary_operation_base<slide_righti_t<V>> {
private:
    friend binary_operation_base<slide_righti_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept
    requires requires {
        { slide_right<V>(internal::abi<A>, lhs, rhs) } -> simd_with_abi<A>;
    }
    {
        return slide_right<V>(internal::abi<A>, lhs, rhs);
    }

public:
    template <simd_type L, simd_with<typename L::value_type> R>
    requires regular_invocable<slide_right_t, L, R, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        return slide_right_t::operator()(lhs, rhs, imm<V>);
    }

    using binary_operation_base<slide_righti_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::slide_lefti_t<V> slide_lefti{};
DPL_EXPORT inline constexpr internal::slide_left_t slide_left{};
DPL_EXPORT template <auto V>
inline constexpr internal::slide_righti_t<V> slide_righti{};
DPL_EXPORT inline constexpr internal::slide_right_t slide_right{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

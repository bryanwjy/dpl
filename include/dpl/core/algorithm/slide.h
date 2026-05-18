// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/reinterpret.h"
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
    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs, size_t num) noexcept {
        constexpr auto simd_size = basic_vector<E, A>::size();
        num = num <= simd_size ? num : simd_size;
        auto const low = dx::shift_left(lhs, num);
        auto const high =
            dx::shift_right(rhs, simd_abi_traits<E, A>::size - num);
        return dx::reinterpret<E>(dx::bwor(low, high));
    }

    template <size_t N, typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        static_assert(N <= simd_abi_traits<E, A>::size);
        constexpr auto size = 2 * simd_abi_traits<E, A>::size;
        auto const low = dx::shift_left(lhs, imm<N>);
        auto const high =
            dx::shift_right(rhs, imm<simd_abi_traits<E, A>::size - N>);
        return dx::reinterpret<E>(dx::bwor(low, high));
    }

public:
    template <simd_vector L, simd_with<typename L::value_type> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_left<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
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
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs, num);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), num);
        }
    }

    template <simd_vector L, simd_with<typename L::value_type> R>
    requires (!same_abi_simd_as<L, R> || scalable_vector<common_abi_t<L, R>>) &&
        (unqualified_slide_left<L, R> ||
            unqualified_slide_left<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_left<L, R>) {
            return slide_left(internal::abi<A>, lhs, rhs, num);
        } else {
            return slide_left(internal::abi<A>, dx::to_canonical(lhs),
                dx::to_canonical(rhs), num);
        }
    }

    template <simd_vector L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_lefti<N, L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
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
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback<N::value>(lhs, rhs);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), num);
        }
    }

    template <simd_vector L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires (!same_abi_simd_as<L, R> || scalable_vector<common_abi_t<L, R>>) &&
        (unqualified_slide_lefti<N, L, R> ||
            unqualified_slide_lefti<N, canonical_type_t<L>,
                canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_left<L, N>) {
            return slide_left<N::value>(internal::abi<A>, lhs, rhs);
        } else {
            return slide_left<N::value>(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
};

struct slide_right_t {
private:
    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs, size_t num) noexcept {
        num = num <= simd_abi_traits<E, A>::size ? num
                                                 : simd_abi_traits<E, A>::size;
        return slide_left_t::operator()(
            lhs, rhs, simd_abi_traits<E, A>::size - num);
    }

    template <size_t N, typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        constexpr auto num = simd_abi_traits<E, A>::size - N;
        return slide_left_t::operator()(lhs, rhs, imm<num>);
    }

public:
    template <simd_vector L, simd_with<typename L::value_type> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_right<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
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
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs, num);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), num);
        }
    }

    template <simd_vector L, simd_with<typename L::value_type> R>
    requires (!same_abi_simd_as<L, R> || scalable_vector<common_abi_t<L, R>>) &&
        (unqualified_slide_right<L, R> ||
            unqualified_slide_right<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_right<L, R>) {
            return slide_right(internal::abi<A>, lhs, rhs, num);
        } else {
            return slide_right(internal::abi<A>, dx::to_canonical(lhs),
                dx::to_canonical(rhs), num);
        }
    }

    template <simd_vector L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_slide_righti<N, L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
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
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallbacki<N::value>(lhs, rhs);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), num);
        }
    }

    template <simd_vector L, simd_with<typename L::value_type> R,
        integral_constant_like N>
    requires (!same_abi_simd_as<L, R> || scalable_vector<common_abi_t<L, R>>) &&
        (unqualified_slide_righti<N, L, R> ||
            unqualified_slide_righti<N, canonical_type_t<L>,
                canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N num) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_slide_right<L, N>) {
            return slide_right<N::value>(internal::abi<A>, lhs, rhs);
        } else {
            return slide_right<N::value>(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
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
    template <simd_vector L, simd_with<typename L::value_type> R>
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
    template <simd_vector L, simd_with<typename L::value_type> R>
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

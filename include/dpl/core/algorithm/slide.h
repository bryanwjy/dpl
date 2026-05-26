// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/operation_base.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t>
void slide_left(...) noexcept = delete;
void slide_left(...) noexcept = delete;
template <size_t>
void slide_right(...) noexcept = delete;
void slide_right(...) noexcept = delete;
struct slide_left_t;
struct slide_right_t;

template <typename T, typename L, typename R>
concept slide_result = simd_vector<T> && simd_vector<L> && simd_vector<R> &&
    same_as<typename L::value_type, typename R::value_type> &&
    common_abi_with<typename L::abi_type, typename R::abi_type> &&
    same_as<typename T::value_type, typename R::value_type> &&
    same_as<typename T::value_type, typename L::value_type> &&
    common_abi_with<typename T::abi_type,
        common_abi_t<typename L::abi_type, typename R::abi_type>>;

template <typename T, typename L, typename R>
concept canonical_slide_result = slide_result<T, L, R> &&
    same_as<typename T::abi_type,
        common_abi_t<typename L::abi_type, typename R::abi_type>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_slide_left =
    requires(L lhs, R rhs, size_t count) {
        {
            slide_left(internal::abi<A>, lhs, rhs, count)
        } -> canonical_slide_result<L, R>;
    };

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_slide_left = requires(L lhs, R rhs, size_t count) {
    { slide_left(lhs, rhs, count) } -> slide_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_left = unqualified_extended_slide_left<L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        regular_invocable<slide_left_t, canonical_type_t<L>,
            canonical_type_t<R>, size_t>);

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_slide_right =
    requires(L lhs, R rhs, size_t count) {
        {
            slide_right(internal::abi<A>, lhs, rhs, count)
        } -> canonical_slide_result<L, R>;
    };

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_slide_right =
    requires(L lhs, R rhs, size_t count) {
        { slide_right(lhs, rhs, count) } -> slide_result<L, R>;
    };

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_right = unqualified_extended_slide_right<L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        regular_invocable<slide_right_t, canonical_type_t<L>,
            canonical_type_t<R>, size_t>);

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_slide_lefti = requires(L lhs, R rhs) {
    {
        slide_left<V::value>(internal::abi<A>, lhs, rhs)
    } -> canonical_slide_result<L, R>;
};

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_slide_lefti = requires(L lhs, R rhs) {
    { slide_left<V::value>(lhs, rhs) } -> slide_result<L, R>;
};

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_lefti =
    unqualified_extended_slide_lefti<V, L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        regular_invocable<slide_left_t, V, canonical_type_t<L>,
            canonical_type_t<R>>);

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_slide_righti = requires(L lhs, R rhs) {
    {
        slide_right<V::value>(internal::abi<A>, lhs, rhs)
    } -> canonical_slide_result<L, R>;
};

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_slide_righti = requires(L lhs, R rhs) {
    { slide_right<V::value>(lhs, rhs) } -> slide_result<L, R>;
};

template <typename V, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_slide_righti =
    unqualified_extended_slide_righti<V, L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        regular_invocable<slide_right_t, V, canonical_type_t<L>,
            canonical_type_t<R>>);

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
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs, size_t count) noexcept {
        if constexpr (unqualified_canonical_slide_left<basic_vector<E, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return fallback(lhs, rhs, count);
            } else {
                return slide_left(internal::abi<A>, lhs, rhs, count);
            }
        } else {
            return fallback(lhs, rhs, count);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> E>
    requires simd_element_for<E, RA> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_slide_left<basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs,
        size_t count) noexcept {
        using A = common_abi_t<LA, RA>;
        return select(internal::abi<A>, lhs, rhs, count);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<typename L::value_type, typename R::value_type> &&
        unqualified_slide_left<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t count) noexcept {
        if constexpr (unqualified_extended_slide_left<L, R>) {
            return select(lhs, rhs, count);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), count);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs, N count) noexcept {
        if constexpr (unqualified_canonical_slide_lefti<N, basic_vector<E, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return fallback(lhs, rhs, count);
            } else {
                return slide_left<N::value>(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs, count);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> E,
        integral_constant_like N>
    requires simd_element_for<E, RA> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_slide_lefti<N, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs, N count) noexcept {
        using A = common_abi_t<LA, RA>;
        return select<N::value>(internal::abi<A>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R, integral_constant_like N>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<typename L::value_type, typename R::value_type> &&
        unqualified_slide_lefti<N, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N count) noexcept {
        if constexpr (unqualified_extended_slide_lefti<N, L, R>) {
            return select<N::value>(lhs, rhs);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), count);
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
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs, size_t count) noexcept {
        if constexpr (unqualified_canonical_slide_right<basic_vector<E, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return fallback(lhs, rhs, count);
            } else {
                return slide_right(internal::abi<A>, lhs, rhs, count);
            }
        } else {
            return fallback(lhs, rhs, count);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> E>
    requires simd_element_for<E, RA> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_slide_right<basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs,
        size_t count) noexcept {
        using A = common_abi_t<LA, RA>;
        return select(internal::abi<A>, lhs, rhs, count);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<typename L::value_type, typename R::value_type> &&
        unqualified_slide_right<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t count) noexcept {
        if constexpr (unqualified_extended_slide_right<L, R>) {
            return select(lhs, rhs, count);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), count);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs, N count) noexcept {
        if constexpr (unqualified_canonical_slide_righti<N, basic_vector<E, A>,
                          basic_vector<E, A>, A>) {
            if consteval {
                return fallback(lhs, rhs, count);
            } else {
                return slide_right<N::value>(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs, count);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> E,
        integral_constant_like N>
    requires simd_element_for<E, RA> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_slide_righti<N, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs, N count) noexcept {
        using A = common_abi_t<LA, RA>;
        return select<N::value>(internal::abi<A>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R, integral_constant_like N>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<typename L::value_type, typename R::value_type> &&
        unqualified_slide_righti<N, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N count) noexcept {
        if constexpr (unqualified_extended_slide_righti<N, L, R>) {
            return select<N::value>(lhs, rhs);
        } else {
            return operator()(
                dx::to_canonical(lhs), dx::to_canonical(rhs), count);
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

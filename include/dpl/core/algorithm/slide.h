// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/internal/operation_base.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
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

template <typename L, typename R, typename N, typename A = common_abi_t<L, R>>
concept unqualified_canonical_slide_lefti = requires(L lhs, R rhs, N count) {
    {
        slide_left(internal::abi<A>, lhs, rhs, count)
    } -> canonical_slide_result<L, R>;
};

template <typename L, typename R, typename N, typename A = common_abi_t<L, R>>
concept unqualified_extended_slide_lefti = requires(L lhs, R rhs, N count) {
    { slide_left(lhs, rhs, count) } -> slide_result<L, R>;
};

template <typename L, typename R, typename N, typename A = common_abi_t<L, R>>
concept unqualified_canonical_slide_righti = requires(L lhs, R rhs, N count) {
    {
        slide_right(internal::abi<A>, lhs, rhs, count)
    } -> canonical_slide_result<L, R>;
};

template <typename L, typename R, typename N, typename A = common_abi_t<L, R>>
concept unqualified_extended_slide_righti = requires(L lhs, R rhs, N count) {
    { slide_right(lhs, rhs, count) } -> slide_result<L, R>;
};

struct slide_left_t {
private:
    template <typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        L lhs, R rhs, size_t num) noexcept {
        using E = simd_element_type_t<L>;
        using A = common_abi_t<L, R>;
        auto simd_size = simd_abi_traits<E, A>::size();
        num = num <= simd_size ? num : simd_size;
        auto const low = dx::shift_left(lhs, num);
        auto const high = dx::shift_right(rhs, simd_size - num);
        return dx::bwor(low, high);
    }

public:
    template <canonical_vector L, canonical_vector R>
    requires same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t count) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<R>>) {
            if constexpr (unqualified_canonical_slide_left<L, R>) {
                if consteval {
                    return slide_left_t::fallback(lhs, rhs, count);
                } else {
                    return slide_left(internal::abi<A>, lhs, rhs, count);
                }
            } else {
                return slide_left_t::fallback(lhs, rhs, count);
            }
        } else if constexpr (unqualified_canonical_slide_left<L, R>) {
            return slide_left(internal::abi<A>, lhs, rhs, count);
        } else {
            return slide_left_t::fallback(lhs, rhs, count);
        }
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t count) noexcept {
        if constexpr (unqualified_extended_slide_left<L, R>) {
            return slide_left(lhs, rhs, count);
        } else if constexpr (simd_expression<L> || simd_expression<R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs), count);
        } else {
            return slide_left_t::fallback(lhs, rhs, count);
        }
    }

    template <canonical_vector L, canonical_vector R, integral_constant_like N>
    requires same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N count) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<R>>) {
            if constexpr (unqualified_canonical_slide_lefti<L, R, N>) {
                if consteval {
                    return operator()(lhs, rhs, N::value);
                } else {
                    return slide_left(internal::abi<A>, lhs, rhs, count);
                }
            } else {
                return operator()(lhs, rhs, N::value);
            }
        } else if constexpr (unqualified_canonical_slide_lefti<L, R, N>) {
            return slide_left(internal::abi<A>, lhs, rhs, count);
        } else {
            return operator()(lhs, rhs, N::value);
        }
    }

    template <simd_vector L, simd_vector R, integral_constant_like N>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N count) noexcept {
        if constexpr (unqualified_extended_slide_lefti<L, R, N>) {
            return slide_left(lhs, rhs, count);
        } else if constexpr (simd_expression<L> || simd_expression<R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs), count);
        } else {
            return operator()(lhs, rhs, N::value);
        }
    }
};

struct slide_right_t {
private:
    template <typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        L lhs, R rhs, size_t num) noexcept {
        using E = simd_element_type_t<L>;
        using A = common_abi_t<L, R>;
        auto const simd_size = simd_abi_traits<E, A>::size();
        num = num <= simd_size ? num : simd_size;
        return slide_left_t::operator()(lhs, rhs, simd_size - num);
    }

public:
    template <canonical_vector L, canonical_vector R>
    requires same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t count) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<R>>) {
            if constexpr (unqualified_canonical_slide_right<L, R>) {
                if consteval {
                    return slide_right_t::fallback(lhs, rhs, count);
                } else {
                    return slide_right(internal::abi<A>, lhs, rhs, count);
                }
            } else {
                return slide_right_t::fallback(lhs, rhs, count);
            }
        } else if constexpr (unqualified_canonical_slide_right<L, R>) {
            return slide_right(internal::abi<A>, lhs, rhs, count);
        } else {
            return slide_right_t::fallback(lhs, rhs, count);
        }
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, size_t count) noexcept {
        if constexpr (unqualified_extended_slide_right<L, R>) {
            return slide_right(lhs, rhs, count);
        } else if constexpr (simd_expression<L> || simd_expression<R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs), count);
        } else {
            return slide_right_t::fallback(lhs, rhs, count);
        }
    }

    template <canonical_vector L, canonical_vector R, integral_constant_like N>
    requires same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N count) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<R>>) {
            if constexpr (unqualified_canonical_slide_righti<L, R, N>) {
                if consteval {
                    return operator()(lhs, rhs, N::value);
                } else {
                    return slide_right(internal::abi<A>, lhs, rhs, count);
                }
            } else {
                return operator()(lhs, rhs, N::value);
            }
        } else if constexpr (unqualified_canonical_slide_righti<L, R, N>) {
            return slide_right(internal::abi<A>, lhs, rhs, count);
        } else {
            return operator()(lhs, rhs, N::value);
        }
    }

    template <simd_vector L, simd_vector R, integral_constant_like N>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs, N count) noexcept {
        if constexpr (unqualified_extended_slide_righti<L, R, N>) {
            return slide_right(lhs, rhs, count);
        } else if constexpr (simd_expression<L> || simd_expression<R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs), count);
        } else {
            return operator()(lhs, rhs, N::value);
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
        {
            slide_left(internal::abi<A>, lhs, rhs, imm<V>)
        } -> simd_type_with_abi<A>;
    }
    {
        return slide_left(internal::abi<A>, lhs, rhs, imm<V>);
    }

public:
    template <simd_vector L, simd_vector R>
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
        {
            slide_right(internal::abi<A>, lhs, rhs, imm<V>)
        } -> simd_type_with_abi<A>;
    }
    {
        return slide_right(internal::abi<A>, lhs, rhs, imm<V>);
    }

public:
    template <simd_vector L, simd_vector R>
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

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rotate_right(...) noexcept = delete;
void rotate_left(...) noexcept = delete;
struct rotate_left_t;
struct rotate_right_t;

template <typename T, typename Arg>
concept rotate_result = simd_vector<T> && simd_vector<Arg> &&
    same_as<typename T::value_type, typename Arg::value_type> &&
    common_abi_with<typename T::abi_type, typename Arg::abi_type>;

template <typename T, typename Arg>
concept canonical_rotate_result = rotate_result<T, Arg> &&
    same_as<typename T::abi_type, typename Arg::abi_type>;

template <typename T>
concept unqualified_canonical_rotate_left = requires(T val, size_t rotate) {
    {
        rotate_left(internal::abi<T>, val, rotate)
    } -> canonical_rotate_result<T>;
};

template <typename T>
concept unqualified_extended_rotate_left = requires(T val, size_t rotate) {
    { rotate_left(val, rotate) } -> rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_canonical_rotate_lefti = requires(T val, N count) {
    { rotate_left(internal::abi<T>, val, count) } -> canonical_rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_rotate_lefti = requires(T val, N count) {
    { rotate_left(val, count) } -> rotate_result<T>;
};

template <typename T>
concept unqualified_canonical_rotate_right = requires(T val, size_t rotate) {
    {
        rotate_right(internal::abi<T>, val, rotate)
    } -> canonical_rotate_result<T>;
};

template <typename T>
concept unqualified_extended_rotate_right = requires(T val, size_t rotate) {
    { rotate_right(val, rotate) } -> rotate_result<T>;
};

template <typename T>
concept unqualified_rotate_right = unqualified_extended_rotate_right<T> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        unqualified_canonical_rotate_right<canonical_type_t<T>>);

template <typename T, typename N>
concept unqualified_canonical_rotate_righti = requires(T val, N count) {
    {
        rotate_right(internal::abi<T>, val, count)
    } -> canonical_rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_rotate_righti = requires(T val, N count) {
    { rotate_right(val, count) } -> rotate_result<T>;
};

struct rotate_right_t {
private:
    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        T val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<T>::size();
        return dx::slide_right(val, val, lanes);
    }

    template <fixed_width_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(T val, N lanes) noexcept {
        constexpr auto V = N::value % simd_abi_traits<T>::size();
        return dx::slide_right(val, val, imm<V>);
    }

public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_rotate_right<T>) {
            if consteval {
                return rotate_right_t::fallback(val, shift);
            } else {
                return rotate_right(internal::abi<T>, val, shift);
            }
        } else {
            return rotate_right_t::fallback(val, shift);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_rotate_right<T>) {
            return rotate_right(val, shift);
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return rotate_right_t::fallback(val, shift);
        }
    }

    template <canonical_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_canonical_rotate_righti<T, N>) {
            if consteval {
                return rotate_right_t::fallback(val, shift);
            } else {
                return rotate_right(internal::abi<T>, val, shift);
            }
        } else {
            return rotate_right_t::fallback(val, shift);
        }
    }

    template <extended_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_rotate_righti<T, N>) {
            return rotate_right(val, shift);
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return rotate_right_t::fallback(val, shift);
        }
    }
};

struct rotate_left_t {
private:
    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        T val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<T>::size();
        return dx::slide_left(val, val, lanes);
    }

    template <fixed_width_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(T val, N lanes) noexcept {
        constexpr auto V = N::value % simd_abi_traits<T>::size();
        return dx::slide_left(val, val, imm<V>);
    }

public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_rotate_left<T>) {
            if consteval {
                return rotate_left_t::fallback(val, shift);
            } else {
                return rotate_left(internal::abi<T>, val, shift);
            }
        } else {
            return rotate_left_t::fallback(val, shift);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_rotate_left<T>) {
            return rotate_left(val, shift);
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return rotate_left_t::fallback(val, shift);
        }
    }

    template <canonical_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_canonical_rotate_lefti<T, N>) {
            if consteval {
                return rotate_left_t::fallback(val, shift);
            } else {
                return rotate_left(internal::abi<T>, val, shift);
            }
        } else {
            return rotate_left_t::fallback(val, shift);
        }
    }

    template <extended_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_rotate_lefti<T, N>) {
            return rotate_left(val, shift);
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return rotate_left_t::fallback(val, shift);
        }
    }
};

template <size_t V>
struct rotate_righti_t {
public:
    template <simd_vector T>
    requires regular_invocable<rotate_right_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return rotate_right_t::operator()(arg, imm<V>);
    }
};

template <size_t V>
struct rotate_lefti_t {
public:
    template <simd_vector T>
    requires regular_invocable<rotate_left_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return rotate_left_t::operator()(arg, imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <size_t V>
inline constexpr internal::rotate_righti_t<V> rotate_righti{};
DPL_EXPORT inline constexpr internal::rotate_right_t rotate_right{};

DPL_EXPORT template <size_t V>
inline constexpr internal::rotate_lefti_t<V> rotate_lefti{};
DPL_EXPORT inline constexpr internal::rotate_left_t rotate_left{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/slide.h"

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
void rotate_right(...) noexcept = delete;
void rotate_right(...) noexcept = delete;
template <size_t>
void rotate_left(...) noexcept = delete;
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

template <typename T>
concept unqualified_rotate_left = unqualified_extended_rotate_left<T> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        unqualified_canonical_rotate_left<canonical_type_t<T>>);

template <typename T, typename N>
concept unqualified_canonical_rotate_lefti = requires(T val) {
    {
        rotate_left<N::value>(internal::abi<T>, val)
    } -> canonical_rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_rotate_lefti = requires(T val) {
    { rotate_left<N::value>(val) } -> rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_rotate_lefti = unqualified_extended_rotate_lefti<T, N> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<rotate_left_t, canonical_type_t<T>, N>);

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
concept unqualified_canonical_rotate_righti = requires(T val) {
    {
        rotate_right<N::value>(internal::abi<T>, val)
    } -> canonical_rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_rotate_righti = requires(T val) {
    { rotate_right<N::value>(val) } -> rotate_result<T>;
};

template <typename T, typename N>
concept unqualified_rotate_righti = unqualified_extended_rotate_righti<T, N> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<rotate_right_t, canonical_type_t<T>, N>);

struct rotate_right_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        lanes %= val.size();
        return dx::slide_right(val, val, lanes);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val, size_t lanes) noexcept {
        lanes %= val.size();
        return dx::slide_right(val, val, lanes);
    }

    template <size_t V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> val) noexcept {
        return dx::slide_righti<V>(val, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t rotate) noexcept {
        if constexpr (unqualified_canonical_rotate_right<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, rotate);
            } else {
                return rotate_right(internal::abi<A>, val, rotate);
            }
        } else {
            return fallback(val, rotate);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_rotate_right<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t rotate) noexcept {
        return rotate_right(internal::abi<A>, val, rotate);
    }

    template <extended_vector T>
    requires unqualified_rotate_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t rotate) noexcept {
        if constexpr (unqualified_extended_rotate_right<T>) {
            return rotate_right(val, rotate);
        } else {
            return operator()(dx::to_canonical(val), rotate);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N rotate) noexcept {
        if constexpr (unqualified_canonical_rotate_righti<basic_vector<E, A>,
                          N>) {
            if consteval {
                return fallbacki<N::value>(val);
            } else {
                return rotate_right<N::value>(internal::abi<A>, val);
            }
        } else {
            return fallbacki<N::value>(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_rotate_righti<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N rotate) noexcept {
        return rotate_right<N::value>(internal::abi<A>, val);
    }

    template <extended_vector T, integral_constant_like N>
    requires unqualified_rotate_righti<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N rotate) noexcept {
        if constexpr (unqualified_extended_rotate_righti<T, N>) {
            return rotate_right<N::value>(val);
        } else {
            return operator()(dx::to_canonical(val), rotate);
        }
    }
};

struct rotate_left_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<E, A>::size;
        return dx::slide_left(val, val, lanes);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<E, A>::size;
        return dx::slide_left(val, val, lanes);
    }

    template <size_t V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> val) noexcept {
        return dx::slide_lefti<V>(val, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t rotate) noexcept {
        if constexpr (unqualified_canonical_rotate_left<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, rotate);
            } else {
                return rotate_left(internal::abi<A>, val, rotate);
            }
        } else {
            return fallback(val, rotate);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_rotate_left<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t rotate) noexcept {
        return rotate_left(internal::abi<A>, val, rotate);
    }

    template <extended_vector T>
    requires unqualified_rotate_left<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t rotate) noexcept {
        if constexpr (unqualified_extended_rotate_left<T>) {
            return rotate_left(val, rotate);
        } else {
            return operator()(dx::to_canonical(val), rotate);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N rotate) noexcept {
        if constexpr (unqualified_canonical_rotate_lefti<basic_vector<E, A>,
                          N>) {
            if consteval {
                return fallbacki<N::value>(val);
            } else {
                return rotate_left<N::value>(internal::abi<A>, val);
            }
        } else {
            return fallbacki<N::value>(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_rotate_lefti<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N rotate) noexcept {
        return rotate_left<N::value>(internal::abi<A>, val);
    }

    template <extended_vector T, integral_constant_like N>
    requires unqualified_rotate_lefti<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N rotate) noexcept {
        if constexpr (unqualified_extended_rotate_lefti<T, N>) {
            return rotate_left<N::value>(val);
        } else {
            return operator()(dx::to_canonical(val), rotate);
        }
    }
};

template <size_t V>
struct rotate_righti_t {
public:
    template <simd_class T>
    requires regular_invocable<rotate_right_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return rotate_right_t::operator()(arg, imm<V>);
    }
};

template <size_t V>
struct rotate_lefti_t {
public:
    template <simd_class T>
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

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t>
void shift_left(...) noexcept = delete;
template <size_t>
void shift_right(...) noexcept = delete;
void shift_left(...) noexcept = delete;
void shift_right(...) noexcept = delete;

struct shift_left_t;
struct shift_right_t;

template <typename T, typename Arg>
concept shift_result = simd_vector<T> && simd_vector<Arg> &&
    same_as<typename T::value_type, typename Arg::value_type> &&
    common_abi_with<typename T::abi_type, typename Arg::abi_type>;

template <typename T, typename Arg>
concept canonical_shift_result = shift_result<T, Arg> &&
    same_as<typename T::abi_type, typename Arg::abi_type>;

template <typename T>
concept unqualified_canonical_shift_left = requires(T val, size_t shift) {
    { shift_left(internal::abi<T>, val, shift) } -> canonical_shift_result<T>;
};

template <typename T>
concept unqualified_extended_shift_left = requires(T val, size_t shift) {
    { shift_left(val, shift) } -> shift_result<T>;
};

template <typename T>
concept unqualified_shift_left = unqualified_extended_shift_left<T> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        unqualified_canonical_shift_left<canonical_type_t<T>>);

template <typename T, typename N>
concept unqualified_canonical_shift_lefti = requires(T val) {
    {
        shift_left<N::value>(internal::abi<T>, val)
    } -> canonical_shift_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_shift_lefti = requires(T val) {
    { shift_left<N::value>(val) } -> shift_result<T>;
};

template <typename T, typename N>
concept unqualified_shift_lefti = unqualified_extended_shift_lefti<T, N> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<shift_left_t, canonical_type_t<T>, N>);

template <typename T>
concept unqualified_canonical_shift_right = requires(T val, size_t shift) {
    { shift_right(internal::abi<T>, val, shift) } -> canonical_shift_result<T>;
};

template <typename T>
concept unqualified_extended_shift_right = requires(T val, size_t shift) {
    { shift_right(val, shift) } -> shift_result<T>;
};

template <typename T>
concept unqualified_shift_right = unqualified_extended_shift_right<T> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        unqualified_canonical_shift_right<canonical_type_t<T>>);

template <typename T, typename N>
concept unqualified_canonical_shift_righti = requires(T val) {
    {
        shift_right<N::value>(internal::abi<T>, val)
    } -> canonical_shift_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_shift_righti = requires(T val) {
    { shift_right<N::value>(val) } -> shift_result<T>;
};

template <typename T, typename N>
concept unqualified_shift_righti = unqualified_extended_shift_righti<T, N> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<shift_right_t, canonical_type_t<T>, N>);

struct shift_left_t {
private:
    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        using traits = simd_abi_traits<E, A>;
        lanes = lanes > traits::size ? traits::size : lanes;
        alignas(traits::alignment) E data[2 * traits::size]{};
        dx::store(val, data);
        return dx::load<E, A>(data + lanes);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_shift_left<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return shift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_shift_left<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return shift_left(internal::abi<A>, val, shift);
    }

    template <extended_vector T>
    requires unqualified_shift_left<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_shift_left<T>) {
            return shift_left(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_shift_lefti<basic_vector<E, A>,
                          N>) {
            if consteval {
                return operator()(val, N::value);
            } else {
                return shift_left<N::value>(internal::abi<A>, val);
            }
        } else {
            return operator()(val, N::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_shift_lefti<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        return shift_left<N::value>(internal::abi<A>, val);
    }

    template <extended_vector T, integral_constant_like N>
    requires unqualified_shift_lefti<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_shift_lefti<T, N>) {
            return shift_left<N::value>(val);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }
};

struct shift_right_t {
private:
    template <typename E, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, size_t lanes) noexcept {
        using traits = simd_abi_traits<E, A>;
        lanes = lanes > traits::size ? traits::size : lanes;
        alignas(traits::alignment) E data[2 * traits::size]{};
        dx::store(val, data + traits::size);
        return dx::load<E, A>(data + traits::size - lanes);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_shift_right<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return shift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_shift_right<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return shift_right(internal::abi<A>, val, shift);
    }

    template <extended_vector T>
    requires unqualified_shift_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_shift_right<T>) {
            return shift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_shift_righti<basic_vector<E, A>,
                          N>) {
            if consteval {
                return operator()(val, N::value);
            } else {
                return shift_right<N::value>(internal::abi<A>, val);
            }
        } else {
            return operator()(val, N::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_shift_righti<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        return shift_right<N::value>(internal::abi<A>, val);
    }

    template <extended_vector T, integral_constant_like N>
    requires unqualified_shift_righti<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_shift_righti<T, N>) {
            return shift_right<N::value>(val);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }
};

template <size_t V>
struct shift_lefti_t {
public:
    template <simd_class T>
    requires regular_invocable<shift_left_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return shift_left_t::operator()(arg, imm<V>);
    }
};

template <size_t V>
struct shift_righti_t {
public:
    template <simd_class T>
    requires regular_invocable<shift_right_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        return shift_right_t::operator()(arg, imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <size_t V>
inline constexpr internal::shift_lefti_t<V> shift_lefti{};
DPL_EXPORT template <size_t V>
inline constexpr internal::shift_righti_t<V> shift_righti{};
DPL_EXPORT inline constexpr internal::shift_left_t shift_left{};
DPL_EXPORT inline constexpr internal::shift_right_t shift_right{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

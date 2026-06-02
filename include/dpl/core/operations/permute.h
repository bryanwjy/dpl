// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/integral_simd.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <size_t...>
void permute(...) noexcept = delete;
template <size_t>
void broadcast_lane(...) noexcept = delete;

template <size_t... Is>
struct permutei_t;

template <typename T, typename Arg>
concept permute_result = simd_vector<T> && simd_vector<Arg> &&
    same_as<typename T::value_type, typename Arg::value_type> &&
    common_abi_with<typename T::abi_type, typename Arg::abi_type>;
template <typename T, typename Arg>
concept canonical_permute_result = permute_result<T, Arg> &&
    same_as<typename T::abi_type, typename Arg::abi_type>;

template <typename T, size_t... Is>
concept unqualified_canonical_permutei = requires(T val) {
    { permute<Is...>(internal::abi<T>, val) } -> canonical_permute_result<T>;
};

template <typename T, size_t... Is>
concept unqualified_extended_permutei = requires(T val) {
    { permute<Is...>(val) } -> permute_result<T>;
};

template <typename T, size_t... Is>
concept unqualified_permutei = unqualified_extended_permutei<T, Is...> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<permutei_t<Is...>, T>);

template <size_t... Is>
struct permutei_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        if constexpr (same_as<index_sequence<Is...>, iota_sequence_t<E, A>>) {
            return arg;
        } else {
            return dx::initialize<E, A>(arg[imm<Is>]...);
        }
    }

    template <typename T>
    static consteval auto extend() noexcept {
        return []<size_t... Js>(index_sequence<Js...>) {
            return permutei_t<Is..., (sizeof...(Is) + Js)...>{};
        }(iota_sequence<T>);
    }

public:
    template <fixed_width_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept
    requires (sizeof...(Is) < simd_abi_traits<T>::size) &&
        regular_invocable<decltype(permutei_t::extend()), T>
    {
        constexpr auto permute = permutei_t::extend();
        return permute(arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires (sizeof...(Is) == simd_abi_traits<A, E>::size)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        static_assert((... && (Is < simd_abi_traits<E, A>::size)));
        if constexpr (unqualified_canonical_permutei<basic_vector<E, A>,
                          Is...>) {
            if consteval {
                return fallback(arg);
            } else {
                return permute<Is...>(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <fixed_width_vector T>
    requires (sizeof...(Is) == simd_abi_traits<T>::size) &&
        unqualified_permutei<T, Is...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        static_assert((... && (Is < simd_abi_traits<T>::size)));
        if constexpr (unqualified_extended_permutei<T, Is...>) {
            return permute<Is...>(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }
};

struct permute_t;

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_canonical_permute = requires(T val, I idx) {
    { permute(internal::abi<A>, val, idx) } -> canonical_permute_result<T>;
};

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_extended_permute = requires(T val, I idx) {
    {
        permute(val, idx)
    } -> permute_result<basic_vector<typename T::value_type, A>>;
};

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_permute = unqualified_extended_permute<T, I, A> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<permute_t, T, I, A>);

struct permute_t {
private:
    template <canonical_class T, common_size_with<simd_lane_type_t<T>> E,
        same_as<typename T::abi_type> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallback(
        T arg, basic_vector<E, A> idx) noexcept {
        return []<size_t... Is>(
                   T arg, basic_vector<E, A> idx, index_sequence<Is...>) {
            return dx::initialize<T>(arg[idx[imm<Is>]]...);
        }(arg, idx, iota_sequence<T>);
    }

public:
    template <simd_class T, size_t... Is>
    requires regular_invocable<permutei_t<Is...>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, index_sequence<Is...>) noexcept {
        constexpr permutei_t<Is...> permute{};
        return permute(arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires common_size_with<E, I> && integral<I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> arg, basic_vector<I, A> idx) noexcept {
        if constexpr (unqualified_canonical_permute<basic_vector<E, A>,
                          basic_vector<I, A>>) {
            if consteval {
                return fallback(arg, idx);
            } else {
                return permute(internal::abi<A>, arg, idx);
            }
        } else {
            return fallback(arg, idx);
        }
    }

    template <simd_abi LA, simd_abi RA, simd_element_for<LA> E,
        simd_element_for<RA> I>
    requires common_size_with<E, I> && integral<I> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_permute<basic_vector<E, RA>, basic_vector<I, LA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, LA> arg, basic_vector<I, RA> idx) noexcept {
        return permute(internal::abi<common_abi_t<LA, RA>>, arg, idx);
    }

    template <simd_vector T, simd_vector I>
    requires (extended_vector<T> || extended_vector<I>) &&
        common_size_with<typename T::value_type, typename I::value_type> &&
        integral<typename I::value_type> && unqualified_permute<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, I idx) noexcept {
        if constexpr (unqualified_extended_permute<T, I>) {
            return permute(arg, idx);
        } else {
            return operator()(dx::to_canonical(arg), dx::to_canonical(idx));
        }
    }
};

template <size_t I>
struct broadcast_lanei_t;

template <typename T, size_t I>
concept unqualified_canonical_broadcast_lanei = requires(T val) {
    { broadcast_lane<I>(internal::abi<T>, val) } -> canonical_permute_result<T>;
};

template <typename T, size_t I>
concept unqualified_extended_broadcast_lanei = requires(T val) {
    { broadcast_lane<I>(val) } -> permute_result<T>;
};

template <typename T, size_t I>
concept unqualified_broadcast_lanei =
    unqualified_extended_broadcast_lanei<T, I> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<broadcast_lanei_t<I>, T>);

template <size_t I>
struct broadcast_lanei_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg) noexcept {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return permutei_t<((Is / Is) * I)...>::operator()(arg);
        }(iota_sequence<E, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires (I < simd_abi_traits<A, E>::size)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> arg) noexcept {
        if constexpr (unqualified_canonical_broadcast_lanei<basic_vector<E, A>,
                          I>) {
            if consteval {
                return fallback(arg);
            } else {
                return broadcast_lane<I>(internal::abi<A>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <fixed_width_vector T>
    requires (I < simd_abi_traits<T>::size) && unqualified_broadcast_lanei<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_extended_broadcast_lanei<T, I>) {
            return broadcast_lane<I>(arg);
        } else {
            return operator()(dx::to_canonical(arg));
        }
    }
};

struct broadcast_lane_t;

template <typename T>
concept unqualified_canonical_broadcast_lane = requires(T val, size_t idx) {
    {
        broadcast_lane(internal::abi<T>, val, idx)
    } -> canonical_permute_result<T>;
};

template <typename T>
concept unqualified_extended_broadcast_lane = requires(T val, size_t idx) {
    { broadcast_lane(val, idx) } -> permute_result<T>;
};

template <typename T>
concept unqualified_broadcast_lane = unqualified_extended_broadcast_lane<T> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<broadcast_lane_t, T>);

struct broadcast_lane_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg, size_t idx) noexcept {
        return dx::broadcast<E, A>(arg[idx]);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t idx) noexcept {
        if constexpr (unqualified_canonical_broadcast_lane<
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, val, idx);
            }
        } else {
            return fallback(val, idx);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_broadcast_lane<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<A>, val, idx);
    }

    template <extended_vector T>
    requires unqualified_broadcast_lane<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t idx) noexcept {
        if constexpr (unqualified_extended_broadcast_lane<T>) {
            return broadcast_lane(val, idx);
        } else {
            return operator()(dx::to_canonical(val), idx);
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <size_t... Is>
inline constexpr internal::permutei_t<Is...> permutei{};
DPL_EXPORT template <size_t I>
inline constexpr internal::broadcast_lanei_t<I> broadcast_lanei{};
DPL_EXPORT inline constexpr internal::permute_t permute{};
DPL_EXPORT inline constexpr internal::broadcast_lane_t broadcast_lane{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

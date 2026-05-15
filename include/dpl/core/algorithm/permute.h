// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/to_basic_type.h"
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

template <typename T, size_t... Is>
concept unqualified_permutei = requires(T val) {
    { permute<Is...>(internal::abi<T>, val) } -> equivalent_class_as<T>;
};

template <typename T, typename I>
concept unqualified_permute = requires(T val, I idx) {
    { permute(internal::abi<T>, val, idx) } -> equivalent_class_as<T>;
};

template <size_t I, typename T>
concept unqualified_broadcast_lanei = requires(T val) {
    { broadcast_lane<I>(internal::abi<T>, val) } -> equivalent_class_as<T>;
};

template <typename T>
concept unqualified_broadcast_lane = requires(T val, size_t idx) {
    { broadcast_lane(internal::abi<T>, val, idx) } -> equivalent_class_as<T>;
};

template <size_t... Is>
struct permutei_t;

template <size_t I>
struct broadcast_lanei_t;

struct permute_t {
private:
    template <basic_simd_class T, common_size_with<simd_lane_type_t<T>> E,
        same_as<typename T::abi_type> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL
        fallback(T arg, basic_simd<E, A> idx) noexcept {
        return []<size_t... Is>(
                   T arg, basic_simd<E, A> idx, index_sequence<Is...>) {
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

    template <simd_class T, integral_simd I>
    requires common_size_simd_with<T, I> && same_abi_simd_as<T, I> &&
        fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, I idx) noexcept {
        if constexpr (unqualified_permute<T, I>) {
            if constexpr (basic_simd_type<T> && basic_simd_type<I>) {
                if consteval {
                    return fallback(arg, idx);
                } else {
                    return permute(internal::abi<T>, arg, idx);
                }
            } else {
                return permute(internal::abi<T>, arg, idx);
            }
        } else if constexpr (basic_simd_type<T> && basic_simd_type<I>) {
            return fallback(arg, idx);
        } else {
            return operator()(dx::to_basic_type(arg), dx::to_basic_type(idx));
        }
    }

    template <simd_class T, integral_simd I>
    requires common_size_simd_with<T, I> && same_abi_simd_as<T, I> &&
        scalable_simd<T> &&
        (unqualified_permute<T, I> ||
            unqualified_permute<basic_type_t<T>, basic_type_t<I>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, I idx) noexcept {
        if constexpr (unqualified_permute<T, I>) {
            return permute(internal::abi<T>, arg, idx);
        } else {
            return permute(internal::abi<T>, dx::to_basic_type(arg),
                dx::to_basic_type(idx));
        }
    }
};

struct broadcast_lane_t {
private:
    template <basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallback(T arg, size_t idx) noexcept {
        return dx::broadcast<simd_lane_type_t<T>>(arg[idx]);
    }

public:
    template <simd_class T, integral_constant_like I>
    requires regular_invocable<broadcast_lanei_t<I::value>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, I idx) noexcept {
        constexpr broadcast_lanei_t<I::value> broadcast_lane{};
        return broadcast_lane(arg);
    }

    template <fixed_width_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, size_t idx) noexcept {
        if constexpr (unqualified_broadcast_lane<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(arg, idx);
                } else {
                    return broadcast_lane(internal::abi<T>, arg, idx);
                }
            } else {
                return broadcast_lane(internal::abi<T>, arg, idx);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg, idx);
        } else {
            return operator()(dx::to_basic_type(arg), idx);
        }
    }

    template <scalable_simd T>
    requires unqualified_broadcast_lane<T> ||
        unqualified_broadcast_lane<basic_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg, size_t idx) noexcept {
        if constexpr (unqualified_broadcast_lane<T>) {
            return broadcast_lane(internal::abi<T>, arg, idx);
        } else {
            return broadcast_lane(
                internal::abi<T>, dx::to_basic_type(arg), idx);
        }
    }
};

template <size_t... Is>
struct permutei_t {
private:
    template <basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallback(T arg) noexcept {
        return dx::initialize<T>(arg[imm<Is>]...);
    }

public:
    template <fixed_width_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        static_assert((sizeof...(Is) <= simd_abi_traits<T>::size) &&
            (... && (Is < simd_abi_traits<T>::size)));
        if constexpr (same_as<index_sequence<Is...>,
                          make_index_sequence<sizeof...(Is)>>) {
            return arg;
        } else if constexpr (sizeof...(Is) < simd_abi_traits<T>::size) {
            return []<size_t... Js>(T arg, index_sequence<Js...>) {
                return permutei_t<Is..., (sizeof...(Is) + Js)...>::operator()(
                    arg);
            }(arg, iota_sequence<T>);
        } else if constexpr (unqualified_permutei<T, Is...>) {
            if constexpr (basic_simd_class<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return permute<Is...>(internal::abi<T>, arg);
                }
            } else {
                return permute<Is...>(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_class<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};

template <size_t I>
struct broadcast_lanei_t {
private:
    template <typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallback(T arg) noexcept {
        return []<size_t... Is>(T arg, index_sequence<Is...>) {
            return permutei_t<((Is / Is) * I)...>::operator()(arg);
        }(arg, iota_sequence<T>);
    }

public:
    template <fixed_width_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        static_assert(I < simd_abi_traits<T>::size);
        if constexpr (unqualified_broadcast_lanei<I, T>) {
            if constexpr (basic_simd_class<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return broadcast_lane<I>(internal::abi<T>, arg);
                }
            } else {
                return broadcast_lane<I>(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_class<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
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

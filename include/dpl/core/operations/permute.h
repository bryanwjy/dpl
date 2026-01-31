// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <auto...>
void permute(...) noexcept = delete;
template <auto>
void broadcast_element(...) noexcept = delete;

template <size_t... Is>
struct permute_t {
private:
    static constexpr bool is_iota =
        same_as<index_sequence<Is...>, make_index_sequence<sizeof...(Is)>>;

    template <basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T fallback(T arg) noexcept
    requires (sizeof...(Is) == element_count<T>)
    {
        return dx::initialize<T>(arg[imm<Is>]...);
    }

public:
    template <basic_simd_class T>
    requires (sizeof...(Is) <= element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (is_iota) {
            return arg;
        } else if constexpr (sizeof...(Is) < element_count<T>) {
            return []<size_t... Js>(T arg, index_sequence<Js...>) {
                return permute_t<Is..., (sizeof...(Is) + Js)...>::operator()(
                    arg);
            }(arg, iota_sequence<T>);
        } else if constexpr (requires {
                                 permute<Is...>(internal::abi<T>, arg);
                             }) {
            if consteval {
                return fallback(arg);
            } else {
                return permute<Is...>(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_class T>
    requires (sizeof...(Is) <= element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> equivalent_class_as<T> auto {
        if constexpr (is_iota) {
            return arg;
        } else if constexpr (sizeof...(Is) < element_count<T>) {
            return []<size_t... Js>(T arg, index_sequence<Js...>) {
                return permute_t<Is..., (sizeof...(Is) + Js)...>::operator()(
                    arg);
            }(arg, iota_sequence<T>);
        } else if constexpr (requires {
                                 permute<Is...>(internal::abi<T>, arg);
                             }) {
            return permute<Is...>(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};

template <size_t I>
struct broadcast_element_t {
private:
    template <basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T fallback(T arg) noexcept {
        return []<size_t... Is>(T arg, index_sequence<Is...>) {
            return permute_t<((Is / Is) * I)...>::operator()(arg);
        }(arg, iota_sequence<T>);
    }

public:
    template <basic_simd_class T>
    requires (I < element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires {
                          broadcast_element<I>(internal::abi<T>, arg);
                      }) {
            if consteval {
                return fallback(arg);
            } else {
                return broadcast_element<I>(internal::abi<T>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_class T>
    requires (I < element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> equivalent_class_as<T> auto {
        if constexpr (requires {
                          broadcast_element<I>(internal::abi<T>, arg);
                      }) {
            return broadcast_element<I>(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <size_t... Is>
inline constexpr internal::permute_t<Is...> permute{};
template <size_t I>
inline constexpr internal::broadcast_element_t<I> broadcast_element{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

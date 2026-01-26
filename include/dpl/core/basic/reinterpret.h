// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/immediate.h"
#include "dpl/core/basic/load.h"
#include "dpl/core/basic/store.h"

#if !DPL_MODULES

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void reinterpret(...) noexcept = delete;

template <typename>
struct reinterpret_t {};

template <simd_element E>
struct reinterpret_t<E> {
public:
    template <basic_simd_class T>
    static constexpr void operator()(T) noexcept = delete;

    template <simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (requires(T arg) {
                          { reinterpret<E>(internal::abi<T>, arg) };
                      }) {
            return reinterpret<E>(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_type T>
    requires same_as<E, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        return arg;
    }

    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<T, E> DPL_VECTORCALL operator()(
        T arg) noexcept {
        if consteval {
            using A = typename T::abi_type;
            static_assert(sizeof(array_for<T>) == sizeof(array_for<E, A>));
            // Cannot bitcast directly due to MSVC using unions as
            // vector types
            array_for<T> from{};
            dx::store(arg, from.data);
            auto const tmp = __DPL bit_cast<array_for<E, A>>(from);
            return dx::load<A>(tmp.data);
        } else {
            if constexpr (requires(T arg) {
                              { reinterpret<E>(internal::abi<T>, arg) };
                          }) {
                return reinterpret<E>(internal::abi<T>, arg);
            } else {
                return __DPL bit_cast<rebind_simd_t<T, E>>(arg);
            }
        }
    }

    template <basic_simd_mask_type T>
    requires same_as<E, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        return arg;
    }

    template <basic_simd_mask_type T>
    requires (element_count<E, typename T::abi_type> == element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<T, E> DPL_VECTORCALL operator()(
        T arg) noexcept {
        if consteval {
            using To = rebind_simd_t<T, E>;
            return []<size_t... Is>(T arg, index_sequence<Is...>) {
                return dx::initialize<To>(dx::extract(arg, imm<Is>)...);
            }(arg, iota_sequence<To>);
        } else {
            if constexpr (requires(T arg) {
                              { reinterpret<E>(internal::abi<T>, arg) };
                          }) {
                return reinterpret<E>(internal::abi<T>, arg);
            } else {
                return __DPL bit_cast<rebind_simd_t<T, E>>(arg);
            }
        }
    }
};

template <typename From, typename To>
concept reinterpretable_as = requires(From from) {
    { reinterpret<To>(internal::abi<From>, from) } -> same_as<To>;
};

template <basic_simd_class To>
struct reinterpret_t<To> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(To src) noexcept { return src; }

    template <simd_class From>
    requires common_class_with<To, From> && simd_same_abi_with<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From from) noexcept {
        return reinterpret_t<simd_element_type_t<To>>::operator()(from);
    }

    template <simd_class From>
    requires common_class_with<To, From> && simd_same_abi_with<From, To> &&
        reinterpretable_as<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From from) noexcept {
        return reinterpret<To>(internal::abi<From>, from);
    }
};

template <simd_class To>
struct reinterpret_t<To> {
private:
    using base_type = reinterpret_t<simd_element_type_t<To>>;

public:
    template <simd_class From>
    requires common_class_with<To, From> && simd_same_abi_with<From, To> &&
        reinterpretable_as<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return reinterpret<To>(internal::abi<From>, arg);
    }

    template <simd_class From>
    requires common_class_with<To, From> && simd_same_abi_with<From, To> &&
        regular_invocable<base_type, From> &&
        explicitly_convertible_to<basic_type_t<To>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return static_cast<To>(base_type::operator()(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <typename To>
inline constexpr internal::reinterpret_t<To> reinterpret{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

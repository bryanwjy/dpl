// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/to_basic_type.h"

#if !DPL_MODULES

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/core/type_traits/rebind_simd.h"
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
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (requires { reinterpret<E>(internal::abi<T>, arg); }) {
            return reinterpret<E>(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_type T>
    requires same_as<E, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T arg) noexcept {
        return arg;
    }

    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<T, E> operator()(T arg) noexcept
    requires requires {
        { reinterpret<E>(internal::abi<T>, arg) };
    }
    {
        return reinterpret<E>(internal::abi<T>, arg);
    }

    template <basic_simd_mask_type T>
    requires same_as<E, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T arg) noexcept {
        return arg;
    }

    template <basic_simd_mask_type T>
    requires (element_count<E, typename T::abi_type> == element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<T, E> operator()(T arg) noexcept
    requires requires {
        { reinterpret<E>(internal::abi<T>, arg) };
    }
    {
        return reinterpret<E>(internal::abi<T>, arg);
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
    requires common_class_with<To, From> && same_abi_simd_as<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From from) noexcept {
        return reinterpret_t<simd_element_type_t<To>>::operator()(from);
    }

    template <simd_class From>
    requires common_class_with<To, From> && same_abi_simd_as<From, To> &&
        reinterpretable_as<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From from) noexcept {
        return reinterpret<To>(internal::abi<From>, from);
    }
};

template <simd_class To>
struct reinterpret_t<To> {
private:
    using base_type DPL_NODEBUG = reinterpret_t<simd_element_type_t<To>>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(To arg) noexcept {
        return arg;
    }

    template <simd_class From>
    requires common_class_with<To, From> && same_abi_simd_as<From, To> &&
        reinterpretable_as<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return reinterpret<To>(internal::abi<From>, arg);
    }

    template <simd_class From>
    requires common_class_with<To, From> && same_abi_simd_as<From, To> &&
        (!reinterpretable_as<From, To> && !equivalent_class_as<To, From>) &&
        regular_invocable<base_type, From> &&
        explicitly_convertible_to<invoke_result_t<base_type, From>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return static_cast<To>(base_type::operator()(arg));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::reinterpret_t<To> reinterpret{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

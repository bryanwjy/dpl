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
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void reinterpret(...) noexcept = delete;

template <typename>
struct reinterpret_t {};

template <typename From, typename To>
concept unqualified_ereinterpretable_as = requires(From from) {
    {
        reinterpret<To>(internal::abi<From>, from)
    } -> simd_with<To, typename From::abi_type>;
};

template <typename From, typename To>
concept unqualified_emreinterpretable_as = requires(From from) {
    {
        reinterpret<To>(internal::abi<From>, from)
    } -> mask_with<To, typename From::abi_type>;
};

template <simd_element E>
struct reinterpret_t<E> {
public:
    template <simd_type T>
    requires (!basic_simd_type<T>) &&
        (unqualified_ereinterpretable_as<T, E> ||
            unqualified_ereinterpretable_as<basic_type_t<T>, E> ||
            same_as<E, simd_element_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (same_as<E, simd_element_type_t<T>>) {
            return arg;
        } else if constexpr (requires {
                                 reinterpret<E>(internal::abi<T>, arg);
                             }) {
            return reinterpret<E>(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_type T>
    requires (unqualified_ereinterpretable_as<T, E> ||
        same_as<E, simd_element_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (same_as<E, simd_element_type_t<T>>) {
            return arg;
        } else {
            return reinterpret<E>(internal::abi<T>, arg);
        }
    }

    template <simd_mask_type T>
    requires (element_count<E, typename T::abi_type> == element_count<T>) &&
        (!basic_simd_mask_type<T>) &&
        (unqualified_emreinterpretable_as<T, E> ||
            unqualified_emreinterpretable_as<basic_type_t<T>, E> ||
            same_as<E, simd_element_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (same_as<E, simd_element_type_t<T>>) {
            return arg;
        } else if constexpr (requires {
                                 reinterpret<E>(internal::abi<T>, arg);
                             }) {
            return reinterpret<E>(internal::abi<T>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <basic_simd_mask_type T>
    requires (element_count<E, typename T::abi_type> == element_count<T>) &&
        (unqualified_emreinterpretable_as<T, E> ||
            same_as<E, simd_element_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (same_as<E, simd_element_type_t<T>>) {
            return arg;
        } else {
            return reinterpret<E>(internal::abi<T>, arg);
        }
    }
};

template <typename From, typename To>
concept unqualified_reinterpretable_as = requires(From from) {
    { reinterpret<To>(internal::abi<From>, from) } -> same_as<To>;
};

template <basic_simd_class To>
struct reinterpret_t<To> {
private:
    using base_type DPL_NODEBUG = reinterpret_t<simd_element_type_t<To>>;

public:
    template <common_class_with<To> From>
    requires (simd_type<From> || element_count<To> == element_count<From>) &&
        same_abi_simd_as<From, To> &&
        (same_as<To, From> || unqualified_reinterpretable_as<From, To> ||
            regular_invocable<base_type, From>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From from) noexcept {
        if constexpr (same_as<To, From>) {
            return from;
        } else if constexpr (unqualified_reinterpretable_as<From, To>) {
            return reinterpret<To>(internal::abi<From>, from);
        } else {
            return reinterpret_t<simd_element_type_t<To>>::operator()(from);
        }
    }
};

template <simd_class To>
struct reinterpret_t<To> {
private:
    using base_type DPL_NODEBUG = reinterpret_t<simd_element_type_t<To>>;

public:
    template <common_class_with<To> From>
    requires (simd_type<From> || element_count<To> == element_count<From>) &&
        same_abi_simd_as<From, To> &&
        (same_as<From, To> || unqualified_reinterpretable_as<From, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        if constexpr (same_as<To, From>) {
            return arg;
        } else {
            return reinterpret<To>(internal::abi<From>, arg);
        }
    }

    template <common_class_with<To> From>
    requires (simd_type<From> || element_count<To> == element_count<From>) &&
        same_abi_simd_as<From, To> &&
        (!unqualified_reinterpretable_as<From, To> && !same_as<From, To>) &&
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

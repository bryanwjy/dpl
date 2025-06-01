#pragma once

#include "dpl/config.h"

#include "dpl/fwd.h"

#include "dpl/blob.h"

#if !DPL_MODULES
#  include <concepts>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T, size_t S>
concept sized_integral = std::integral<T> && sizeof(T) == S;

template <typename T, size_t S>
concept sized_unsigned_integral = std::unsigned_integral<T> && sizeof(T) == S;

template <typename T, size_t S>
concept sized_signed_integral = std::signed_integral<T> && sizeof(T) == S;

template <typename T, size_t S>
concept sized_floating_point = std::floating_point<T> && sizeof(T) == S;

template <typename T, size_t S>
concept sized_blob = (S > 8) && std::same_as<blob<S>, T>;

template <typename T>
concept simd_element =
    sized_integral<T, 1> || sized_integral<T, 2> || sized_integral<T, 4> ||
    sized_integral<T, 8> || sized_floating_point<T, 4> ||
    sized_floating_point<T, 8> || sized_integral<T, 16> || sized_blob<T, 16> ||
    sized_blob<T, 32> || sized_blob<T, 64>;

template <typename T>
concept abi_tag = std::is_empty_v<T> && std::semiregular<T> &&
    is_abi_tag_v<T> && requires { requires T::size > 0; };

DPL_DEFAULT_NAMESPACE_END

DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept has_simd_member_types = requires {
    typename T::element_type;
    typename T::native_type;
    requires simd_element<typename T::element_type>;
    typename native_type_t<typename T::element_type>;
};

template <typename T>
concept convertible_to_native_simd = has_simd_member_types<T> &&
    std::convertible_to<T, typename T::native_type> && requires(T val) {
        { +val } -> std::same_as<typename T::native_type>;
    };

DPL_DEFAULT_NAMESPACE_END

DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept simd_type = std::semiregular<T> && is_simd_type_v<T> &&
    has_simd_member_types<T> && convertible_to_native_simd<T>;

template <typename T>
concept simd_mask_type = std::semiregular<T> && is_simd_mask_type_v<T> &&
    has_simd_member_types<T> && convertible_to_native_simd<T>;

template <typename T, typename E>
concept simd_type_for =
    simd_type<T> && std::same_as<typename T::element_type, E>;

template <typename T, typename E>
concept simd_mask_type_for =
    simd_mask_type<T, E> && std::same_as<typename T::element_type, E>;

template <typename T>
concept scalar_element_type = std::floating_point<T> && simd_element<T>;

DPL_DEFAULT_NAMESPACE_END

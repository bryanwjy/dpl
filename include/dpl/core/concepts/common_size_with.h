// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/common_size_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
namespace atom {
template <typename T, typename U>
concept common_size_with =
    sizeof(T) == sizeof(U) && sizeof(common_size_type_t<T, U>) == sizeof(T) &&
    sizeof(common_size_type_t<T, U>) == sizeof(U);
}

template <typename T, typename U>
concept common_size_with =
    requires {
        typename common_size_type_t<T, U>;
        typename common_size_type_t<U, T>;
    } && same_as<common_size_type_t<U, T>, common_size_type_t<T, U>> &&
    atom::common_size_with<T, U> && atom::common_size_with<U, T>;

namespace internal {

template <typename... Ts>
inline constexpr bool is_common_size_with = false;

template <typename A>
inline constexpr bool is_common_size_with<A> = true;

template <typename A, typename B>
inline constexpr bool is_common_size_with<A, B> = common_size_with<A, B>;

template <typename A, typename B, typename... Cs>
requires common_size_with<A, B>
inline constexpr bool is_common_size_with<A, B, Cs...> =
    is_common_size_with<common_size_type_t<A, B>, Cs...>;

template <typename... Ts>
concept all_common_size = internal::is_common_size_with<Ts...>;

} // namespace internal

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

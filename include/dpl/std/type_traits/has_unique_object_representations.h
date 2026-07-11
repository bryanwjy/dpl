// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(has_unique_object_representations)
template <typename T>
inline constexpr bool has_unique_object_representations_v =
    __has_unique_object_representations(T);
template <typename T>
struct has_unique_object_representations :
    bool_constant<__has_unique_object_representations(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(has_unique_object_representations)
template <typename T>
inline constexpr bool has_unique_object_representations_v =
    details::type_traits::unsupported_trait_v<T>;
template <typename T>
struct has_unique_object_representations :
    details::type_traits::unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(has_unique_object_representations)

__DPL_DEFAULT_NAMESPACE_END

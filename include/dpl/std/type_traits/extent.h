// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(array_extent)
template <typename T, size_t D = 0>
inline constexpr size_t extent_v = __array_extent(T, D);

template <typename T, size_t D = 0>
struct extent : size_constant<__array_extent(T, D)> {};

#else // if __UTL_SHOULD_USE_BUILTIN(array_extent)

template <typename T, size_t D>
inline constexpr size_t extent_v = 0;
template <typename T>
inline constexpr size_t extent_v<T[], 0> = 0;
template <typename T, size_t D>
inline constexpr size_t extent_v<T[], D> = extent_v<T, D - 1>;
template <typename T, size_t N>
inline constexpr size_t extent_v<T[N], 0> = N;
template <typename T, size_t N, size_t D>
inline constexpr size_t extent_v<T[N], D> = extent_v<T, D - 1>;
template <typename T>
struct extent : size_constant<extent_v<T>> {};

#endif // if __UTL_SHOULD_USE_BUILTIN(array_extent)

__DPL_DEFAULT_NAMESPACE_END

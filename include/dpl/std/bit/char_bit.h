// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/popcount.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

inline constexpr size_t char_bit_v =
    __DPL popcount(static_cast<unsigned char>(-1));

template <typename T>
inline constexpr size_t type_bit_v = sizeof(T) * __DPL char_bit_v;

template <typename T>
inline constexpr int unsigned_width_diff_v =
    char_bit_v * ((int)sizeof(unsigned) - (int)sizeof(T));

__DPL_DEFAULT_NAMESPACE_END

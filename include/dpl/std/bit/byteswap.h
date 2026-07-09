// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/bit_type.h"
#include "dpl/std/bit/char_bit.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <integral T>
requires requires(T val) { __DPL details::bit::byteswap(val); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return __DPL details::bit::byteswap(val);
}

DPL_DEFAULT_NAMESPACE_END

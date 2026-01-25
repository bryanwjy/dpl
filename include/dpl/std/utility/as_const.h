// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
void as_const(T&&) noexcept = delete;

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(NODISCARD, ALWAYS_INLINE, _HIDE_FROM_ABI)
constexpr T const& as_const(T const& value DPL_LIFETIMEBOUND) noexcept {
    return value;
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(NODISCARD, ALWAYS_INLINE, _HIDE_FROM_ABI)
constexpr T const& as_const(T& value DPL_LIFETIMEBOUND) noexcept {
    return value;
}

DPL_DEFAULT_NAMESPACE_END

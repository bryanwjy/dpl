// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

DPL_ATTRIBUTES(_HIDE_FROM_ABI, NORETURN)
inline void unreachable() noexcept {
    DPL_BUILTIN_unreachable();
}

__DPL_DEFAULT_NAMESPACE_END

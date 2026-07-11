// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

struct template_barrier_t {
    __DPL_HIDE_FROM_ABI explicit constexpr template_barrier_t() noexcept =
        default;
};

inline constexpr template_barrier_t template_barrier{};

__DPL_DEFAULT_NAMESPACE_END

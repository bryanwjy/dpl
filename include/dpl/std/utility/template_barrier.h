// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT struct template_barrier_t {
    __DPL_HIDE_FROM_ABI explicit constexpr template_barrier_t() noexcept =
        default;
};

DPL_EXPORT inline constexpr template_barrier_t template_barrier{};

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT struct loading_t {
    explicit constexpr loading_t() noexcept = default;
};
DPL_EXPORT inline constexpr loading_t loading{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

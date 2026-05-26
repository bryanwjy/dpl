// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT struct broadcasting_t {
    explicit constexpr broadcasting_t() noexcept = default;
};
DPL_EXPORT inline constexpr broadcasting_t broadcasting{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT struct indices_t {
    explicit constexpr indices_t() noexcept = default;
};
DPL_EXPORT inline constexpr indices_t indices{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

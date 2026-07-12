// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
struct broadcasting_t {
    explicit constexpr broadcasting_t() noexcept = default;
};
inline constexpr broadcasting_t broadcasting{};
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

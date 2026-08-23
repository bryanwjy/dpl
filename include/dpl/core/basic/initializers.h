// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
struct unspecified_t {
    explicit constexpr unspecified_t() noexcept = default;
};
inline constexpr unspecified_t unspecified{};

struct broadcasting_t {
    explicit constexpr broadcasting_t() noexcept = default;
};
inline constexpr broadcasting_t broadcasting{};

struct iota_t {
    explicit constexpr iota_t() noexcept = default;
};
inline constexpr iota_t iota{};
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

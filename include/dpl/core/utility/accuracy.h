// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT struct max_accuracy_t {
    __DPL_HIDE_FROM_ABI explicit constexpr max_accuracy_t() noexcept = default;
};
DPL_EXPORT struct approximate_t {
    __DPL_HIDE_FROM_ABI explicit constexpr approximate_t() noexcept = default;
};

DPL_EXPORT inline constexpr max_accuracy_t max_accuracy{};
DPL_EXPORT inline constexpr approximate_t approximate{};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

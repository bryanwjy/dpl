// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT
struct aligned_t {
    __DPL_HIDE_FROM_ABI explicit constexpr aligned_t() noexcept = default;
};

DPL_EXPORT inline constexpr aligned_t aligned{};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

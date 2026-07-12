// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
struct aligned_t {
    __DPL_HIDE_FROM_ABI explicit constexpr aligned_t() noexcept = default;
};

inline constexpr aligned_t aligned{};
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

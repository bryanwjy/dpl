// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT struct unsafe_t {
    __DPL_HIDE_FROM_ABI explicit constexpr unsafe_t() noexcept = default;
};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

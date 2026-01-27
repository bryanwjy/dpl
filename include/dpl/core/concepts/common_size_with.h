// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <typename T, typename U>
concept common_size_with = sizeof(T) == sizeof(U);
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

// Cheaper declval, assumes T is always referenceable
template <typename T>
T&& declarg() noexcept;

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END

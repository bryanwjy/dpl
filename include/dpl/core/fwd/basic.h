// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef.h" // IWYU pragma: export
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename E, typename A>
class basic_vector;
DPL_EXPORT template <typename E, typename A>
class basic_mask;
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

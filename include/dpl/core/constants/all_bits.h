// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/all_bits.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT inline constexpr all_bits_t all_bits{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<all_bits_t, T>
inline constexpr auto all_bits_v = static_cast<T>(all_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

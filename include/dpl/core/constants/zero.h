// Copyright 2025-2026 Bryan Wong
#pragma once

#if !DPL_MODULES
#  include "dpl/core/basic/zero.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT inline constexpr zero_t zero{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<zero_t, T>
inline constexpr auto zero_v = static_cast<T>(zero);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

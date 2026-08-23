// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/operations/compare.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
inline namespace cpo {
inline constexpr internal::cmplt_t isless{};
inline constexpr internal::cmpgt_t isgreater{};
inline constexpr internal::cmple_t islessequal{};
inline constexpr internal::cmpge_t isgreaterequal{};
inline constexpr internal::cmpunord_t isunordered{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

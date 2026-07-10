// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/math/details/rounding.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/utility/to_underlying.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace internal {
void round(...) noexcept = delete;
}

DPL_EXPORT namespace datapar::rounding {
using to_nearest_int_t = mx::rounding_t<mx::rounding_flags::to_nearest_int>;
using to_pos_inf_t = mx::rounding_t<mx::rounding_flags::to_pos_inf>;
using to_neg_inf_t = mx::rounding_t<mx::rounding_flags::to_neg_inf>;
using to_zero_t = mx::rounding_t<mx::rounding_flags::to_zero>;
using current_dir_t = mx::rounding_t<mx::rounding_flags::current_dir>;
using no_exc_t = mx::rounding_t<mx::rounding_flags::no_exc>;
inline constexpr mx::rounding_t<mx::rounding_flags::to_nearest_int>
    to_nearest_int{};
inline constexpr mx::rounding_t<mx::rounding_flags::to_pos_inf> to_pos_inf{};
inline constexpr mx::rounding_t<mx::rounding_flags::to_neg_inf> to_neg_inf{};
inline constexpr mx::rounding_t<mx::rounding_flags::to_zero> to_zero{};
inline constexpr mx::rounding_t<mx::rounding_flags::current_dir> current_dir{};
inline constexpr mx::rounding_t<mx::rounding_flags::no_exc> no_exc{};
} // namespace datapar::rounding

DPL_EXPORT namespace rounding = __DPL datapar::rounding; // NOLINT
DPL_DEFAULT_NAMESPACE_END

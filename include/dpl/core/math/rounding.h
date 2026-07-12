// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/rounding.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace internal {
void round(...) noexcept = delete;
}

namespace datapar::rounding {
using to_nearest_int_t =
    fmath::rounding_t<fmath::rounding_flags::to_nearest_int>;
using to_pos_inf_t = fmath::rounding_t<fmath::rounding_flags::to_pos_inf>;
using to_neg_inf_t = fmath::rounding_t<fmath::rounding_flags::to_neg_inf>;
using to_zero_t = fmath::rounding_t<fmath::rounding_flags::to_zero>;
using current_dir_t = fmath::rounding_t<fmath::rounding_flags::current_dir>;
using no_exc_t = fmath::rounding_t<fmath::rounding_flags::no_exc>;
inline constexpr fmath::rounding_t<fmath::rounding_flags::to_nearest_int>
    to_nearest_int{};
inline constexpr fmath::rounding_t<fmath::rounding_flags::to_pos_inf>
    to_pos_inf{};
inline constexpr fmath::rounding_t<fmath::rounding_flags::to_neg_inf>
    to_neg_inf{};
inline constexpr fmath::rounding_t<fmath::rounding_flags::to_zero> to_zero{};
inline constexpr fmath::rounding_t<fmath::rounding_flags::current_dir>
    current_dir{};
inline constexpr fmath::rounding_t<fmath::rounding_flags::no_exc> no_exc{};
} // namespace datapar::rounding

namespace rounding = __DPL datapar::rounding; // NOLINT
__DPL_DEFAULT_NAMESPACE_END

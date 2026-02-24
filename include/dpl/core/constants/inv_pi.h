// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT
struct inv_pi_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr inv_pi_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this inv_pi_t) noexcept {
        return 0.318309886183790671537767526745028;
    }
};

DPL_EXPORT
inline constexpr inv_pi_t inv_pi{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<inv_pi_t, T>
inline constexpr auto inv_pi_v = static_cast<T>(inv_pi);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct inv_pi_t : broadcastable_base<inv_pi_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr inv_pi_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this inv_pi_t) noexcept {
        return static_cast<T>(0.318309886183790671537767526745028);
    }
};

inline constexpr inv_pi_t inv_pi{};

template <typename T>
requires explicitly_convertible_to<inv_pi_t, T>
inline constexpr T inv_pi_v = static_cast<T>(inv_pi);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

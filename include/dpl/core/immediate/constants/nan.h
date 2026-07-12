// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/all_bits.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct nan_t : broadcastable_base<nan_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr nan_t() noexcept = default;

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this nan_t) noexcept {
        return all_bits_v<T>;
    }
};

inline constexpr nan_t nan{};

template <typename T>
requires explicitly_convertible_to<nan_t, T>
inline constexpr auto nan_v = static_cast<T>(nan);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

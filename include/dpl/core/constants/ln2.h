// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct ln2_t : broadcastable_base<ln2_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr ln2_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ln2_t) noexcept {
        return 0.693147180559945309417232121458176;
    }
};

DPL_EXPORT inline constexpr ln2_t ln2{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<ln2_t, T>
inline constexpr auto ln2_v = static_cast<T>(ln2);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

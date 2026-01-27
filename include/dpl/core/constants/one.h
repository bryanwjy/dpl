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
struct one_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr one_t() noexcept = default;

    template <simd_element T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this one_t) noexcept {
        return static_cast<T>(1);
    }
};

DPL_EXPORT
inline constexpr one_t one{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<one_t, T>
inline constexpr auto one_v = static_cast<T>(one);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

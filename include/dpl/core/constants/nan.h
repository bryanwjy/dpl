// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/all_bits.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT
struct nan_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr nan_t() noexcept = default;

    template <simd_element T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this nan_t) noexcept {
        return all_bits_v<T>;
    }
};

DPL_EXPORT
inline constexpr nan_t nan{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<nan_t, T>
inline constexpr auto nan_v = static_cast<T>(nan);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

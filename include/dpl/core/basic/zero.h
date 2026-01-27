// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_element.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT
struct zero_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr zero_t() noexcept = default;

    template <simd_element T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this zero_t) noexcept {
        return static_cast<T>(0);
    }
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT
struct all_bits_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr all_bits_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this all_bits_t) noexcept {
        return static_cast<T>(-1);
    }

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this all_bits_t self) noexcept {
        if constexpr (sizeof(T) == sizeof(int)) {
            return __DPL bit_cast<T>(static_cast<int>(self));
        } else if constexpr (sizeof(T) == sizeof(int64)) {
            return __DPL bit_cast<T>(static_cast<int64>(self));
        } else {
            static_assert(sizeof(T) == sizeof(short));
            return __DPL bit_cast<T>(static_cast<short>(self));
        }
    }
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

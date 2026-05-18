// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT struct all_bits_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr all_bits_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this all_bits_t) noexcept {
        return static_cast<T>(-1);
    }

    template <typename T>
    requires (!integral<T> && common_bits_with<unsigned_representation_t<T>, T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this all_bits_t self) noexcept {
        return __DPL bit_cast<T>(
            static_cast<unsigned_representation_t<T>>(self));
    }
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

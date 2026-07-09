// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if DPL_HAS_BUILTIN(__builtin_clzg)
#  define __DPL_clz(...)                                             \
      [](auto arg) {                                                 \
        return arg ? __builtin_clzg(arg) : sizeof(arg) * char_bit_v; \
      }(__VA_ARGS__)
#else
#  define __DPL_clz(...) __DPL details::bit::clz(__VA_ARGS__)
#endif

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_zero(T val) noexcept {
    return __DPL_clz(val);
}

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_one(T val) noexcept {
    return __DPL_clz(static_cast<T>(~val));
}

#undef __DPL_clz
DPL_DEFAULT_NAMESPACE_END

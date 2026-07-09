// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/char_bit.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/details/bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if DPL_HAS_BUILTIN(__builtin_ctzg)

#  define __DPL_ctz(...)                                             \
      [](auto arg) {                                                 \
        return arg ? __builtin_ctzg(arg) : sizeof(arg) * char_bit_v; \
      }(__VA_ARGS__)
#else
#  define __DPL_ctz(...) __DPL details::bit::ctz(__VA_ARGS__)
#endif

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_zero(T val) noexcept {

    return __DPL_ctz(val);
}

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_one(T val) noexcept {
    return __DPL_ctz(static_cast<T>(~val));
}

#undef __DPL_ctz
DPL_DEFAULT_NAMESPACE_END

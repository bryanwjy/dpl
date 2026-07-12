// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct one_t : broadcastable_base<one_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr one_t() noexcept = default;

    template <typename T>
    requires requires { static_cast<T>(1); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this one_t) noexcept {
        return static_cast<T>(1);
    }
};

inline constexpr one_t one{};

template <typename T>
requires explicitly_convertible_to<one_t, T>
inline constexpr auto one_v = static_cast<T>(one);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

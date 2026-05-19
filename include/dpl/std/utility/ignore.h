// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT struct ignore_t {
    explicit consteval ignore_t() noexcept = default;

    template <typename T>
    requires different_from<remove_cvref_t<T>, ignore_t>
    __DPL_HIDE_FROM_ABI constexpr void operator=(this ignore_t, T&&) noexcept {}
};

DPL_EXPORT inline constexpr ignore_t ignore{};

DPL_DEFAULT_NAMESPACE_END

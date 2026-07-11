// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_invocable.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename F, typename... Args>
concept invocable = requires(F&& func, Args&&... args) {
    __DPL invoke(static_cast<F&&>(func), static_cast<Args&&>(args)...);
};

template <typename F, typename... Args>
concept regular_invocable = invocable<F, Args...>;

__DPL_DEFAULT_NAMESPACE_END

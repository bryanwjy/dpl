// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::type_traits {
template <typename T>
__DPL_HIDE_FROM_ABI T&& declval(int) noexcept;

template <typename T>
__DPL_HIDE_FROM_ABI T declval(float) noexcept;
} // namespace details::type_traits

template <typename T>
__DPL_HIDE_FROM_ABI auto declval() noexcept
    -> decltype(__DPL details::type_traits::declval<T>(0));

__DPL_DEFAULT_NAMESPACE_END

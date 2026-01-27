// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details {
template <typename T>
__DPL_HIDE_FROM_ABI T&& declval(int) noexcept;

template <typename T>
__DPL_HIDE_FROM_ABI T declval(float) noexcept;
} // namespace details

DPL_EXPORT template <typename T>
__DPL_HIDE_FROM_ABI auto declval() noexcept
    -> decltype(__DPL details::declval<T>(0));

DPL_DEFAULT_NAMESPACE_END

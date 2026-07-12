// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/evaluate.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto) DPL_VECTORCALL forward_or_eval(T&& val) noexcept {
    return static_cast<T&&>(val);
}

template <simd_expression T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto DPL_VECTORCALL forward_or_eval(T&& val) noexcept {
    return dx::evaluate(__DPL forward<T>(val));
}

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END

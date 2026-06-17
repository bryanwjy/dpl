// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/private/concepts.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename D, typename... Ts>
concept extension_fallback = extended_arguments<Ts...> &&
    (... && simd_expression<Ts>) && fallback_cpo_invocable<D, Ts...>;
} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END

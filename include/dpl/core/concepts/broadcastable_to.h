// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <typename T>
struct broadcast_t;
}

DPL_EXPORT template <typename T, typename U>
concept broadcastable_to = (simd_class<U> || simd_abi<U>) &&
    regular_invocable<internal::broadcast_t<U>, T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

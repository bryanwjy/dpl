// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/cpo_invocable.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <typename, typename = ignore_t>
struct broadcast_t;
}

DPL_EXPORT template <typename T, typename U>
concept broadcastable_to = !simd_type<T> && (simd_type<U> || simd_abi<U>) &&
    internal::cpo_invocable<internal::broadcast_t<U>, T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

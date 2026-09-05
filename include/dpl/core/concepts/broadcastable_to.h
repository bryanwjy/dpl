// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/cpo_invocable.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <typename, typename = void>
struct broadcast_t;
}

template <typename T, typename U>
concept broadcastable_to = !simd_type<T> && (simd_type<U> || simd_abi<U>) &&
    internal::cpo_invocable<internal::broadcast_t<U>, T> &&
    noexcept(internal::broadcast_t<U>{}(internal::declarg<T>()));

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

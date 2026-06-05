// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <simd_abi A>
consteval remove_cv_t<A> make_abi() noexcept {
    return remove_cv_t<A>{};
}

template <simd_type T>
consteval simd_abi_type_t<T> make_abi() noexcept {
    return simd_abi_type_t<T>{};
}

template <typename A>
requires requires { internal::make_abi<A>(); }
inline constexpr auto abi = make_abi<A>();
} // namespace internal
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

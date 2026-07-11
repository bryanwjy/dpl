// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/fwd.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept has_simd_abi = requires { typename simd_abi_type<T>::type; };

template <typename T>
concept has_simd_element = requires { typename simd_element_type<T>::type; };

template <typename T>
concept has_simd_members = has_simd_element<T> && has_simd_abi<T>;
} // namespace datapar::internal
__DPL_DEFAULT_NAMESPACE_END

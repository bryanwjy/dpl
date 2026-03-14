// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_class_with.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename A, typename B>
concept common_size_with = sizeof(A) == sizeof(B);

DPL_EXPORT template <typename A, typename B>
concept common_size_simd_with = common_class_with<A, B> &&
    atom::common_abi_with<typename A::abi_type, typename B::abi_type> &&
    common_size_with<simd_element_type_t<A>, simd_element_type_t<B>>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

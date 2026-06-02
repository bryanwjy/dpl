// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct abi_type {};

DPL_EXPORT template <typename T>
using abi_type_t = typename abi_type<T>::type;

DPL_EXPORT template <simd_class T>
struct abi_type<T> {
    using type DPL_NODEBUG = typename T::abi_type;
};

DPL_EXPORT template <simd_abi T>
struct abi_type<T> {
    using type DPL_NODEBUG = T;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

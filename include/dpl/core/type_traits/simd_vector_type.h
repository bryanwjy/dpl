// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/concepts/simd_vector.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_vector_type {};

DPL_EXPORT template <typename T>
using simd_vector_type_t = typename simd_vector_type<T>::type;

DPL_EXPORT template <simd_vector T>
struct simd_vector_type<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <simd_mask T>
struct simd_vector_type<T> {
    using type DPL_NODEBUG = typename T::vector_type;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

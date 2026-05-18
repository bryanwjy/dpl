// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_lane_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_class.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct make_simd_type {};

DPL_EXPORT template <simd_class T>
using make_simd_type_t = typename make_simd_type<T>::type;

DPL_EXPORT template <simd_class T>
requires simd_vector<T>
struct make_simd_type<T> {
    using type DPL_NODEBUG = T;
};

/**
 * Non-basic types may specialize this class to return the corresponding
 * simd_vector. If the returned type does not satisfy simd_vector, behaviour
 * is undefined.
 */
DPL_EXPORT template <simd_class T>
struct make_simd_type<T> {
    using type DPL_NODEBUG =
        basic_vector<simd_lane_type_t<T>, typename T::abi_type>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/concepts/simd_vector.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * Specializable trait to rebind a simd class's element abi type
 *
 * It is undefined behaviour to define specializations (partial or otherwise)
 * to rebind basic classes, i.e. basic_vector, basic_mask.
 */
DPL_EXPORT template <simd_class T, typename U, typename A = typename T::abi_type>
struct rebind_simd {};

DPL_EXPORT template <simd_class T, typename E, simd_abi A = typename T::abi_type>
requires simd_element_for<E, A>
using rebind_simd_t = typename rebind_simd<T, E, A>::type;

DPL_EXPORT template <simd_class T, typename E, simd_abi A>
requires simd_vector<T> && simd_element_for<E, A>
struct rebind_simd<T, E, A> {
    using type DPL_NODEBUG = basic_vector<E, A>;
};

DPL_EXPORT template <simd_class T, typename E, simd_abi A>
requires simd_mask<T> && simd_element_for<E, A>
struct rebind_simd<T, E, A> {
    using type DPL_NODEBUG = basic_mask<E, A>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_abi.h"
#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_abi_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * Specializable trait to rebind a simd class's element abi type
 *
 * It is undefined behaviour to define specializations (partial or otherwise)
 * to rebind basic classes, i.e. basic_vector, basic_mask.
 */
template <typename T, typename E, typename A = simd_abi_type_t<T>>
struct rebind_simd {};

template <typename T, typename E, typename A = simd_abi_type_t<T>>
using rebind_simd_t = typename rebind_simd<T, E, A>::type;

template <typename T, typename E, typename A>
requires enable_simd_vector<T> && enable_simd_abi<A>
struct rebind_simd<T, E, A> {
    using type DPL_NODEBUG = basic_vector<E, A>;
};

template <typename T, typename E, typename A>
requires enable_simd_mask<T> && enable_simd_abi<A>
struct rebind_simd<T, E, A> {
    using type DPL_NODEBUG = basic_mask<E, A>;
};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

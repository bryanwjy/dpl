// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_class_with.h"
#include "dpl/core/concepts/simd_lane_type.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T, typename U>
concept equivalent_class_as = common_class_with<T, U> &&
    same_as<simd_lane_type_t<T>, simd_lane_type_t<U>> &&
    same_as<typename T::abi_type, typename U::abi_type>;

DPL_EXPORT template <typename T, typename U>
concept equivalent_simd_as =
    simd_vector<T> && simd_vector<U> && equivalent_class_as<T, U>;

DPL_EXPORT template <typename T, typename U>
concept equivalent_mask_as =
    simd_mask<T> && simd_mask<U> && equivalent_class_as<T, U>;

DPL_EXPORT template <typename T, typename E, typename A = typename T::abi_type>
concept simd_with = simd_abi<A> && simd_element_for<E, A> && simd_vector<T> &&
    equivalent_simd_as<T, basic_vector<E, A>>;

DPL_EXPORT template <typename T, typename E, typename A = typename T::abi_type>
concept mask_with = simd_abi<A> && simd_element_for<E, A> && simd_mask<T> &&
    equivalent_mask_as<T, basic_mask<E, A>>;

DPL_EXPORT template <typename T, typename A>
concept simd_with_abi =
    atom::simd_basics<T> && simd_with<T, typename T::value_type, A>;

template <typename T, typename E, typename A = typename T::abi_type>
concept vector_with = simd_with<T, E, A>;
template <typename T, typename A>
concept vector_with_abi = simd_vector<T> && same_as<A, typename T::abi_type>;
template <typename T, typename A>
concept mask_with_abi = simd_mask<T> && same_as<A, typename T::abi_type>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

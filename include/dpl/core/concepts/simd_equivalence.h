// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_class_with.h"
#include "dpl/core/concepts/simd_mask_type.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/core/type_traits/simd_lane_type.h"
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
    simd_type<T> && simd_type<U> && equivalent_class_as<T, U>;

DPL_EXPORT template <typename T, typename U>
concept equivalent_mask_as =
    simd_mask_type<T> && simd_mask_type<U> && equivalent_class_as<T, U>;

DPL_EXPORT template <typename T, typename E, typename A = typename T::abi_type>
concept simd_with = simd_element<E> && simd_abi<A> && simd_type<T> &&
    equivalent_simd_as<T, basic_simd<E, A>>;

DPL_EXPORT template <typename T, typename E, typename A = typename T::abi_type>
concept mask_with = simd_element<E> && simd_abi<A> && simd_mask_type<T> &&
    equivalent_mask_as<T, basic_simd_mask<E, A>>;

DPL_EXPORT template <typename T, typename A>
concept simd_with_abi =
    atom::simd_basics<T> && simd_with<T, typename T::value_type, A>;

DPL_EXPORT template <typename T, typename A>
concept mask_with_abi =
    atom::simd_basics<T> && mask_with<T, simd_lane_type_t<T>, A>;

DPL_EXPORT template <typename T, typename A>
concept simd_class_with_abi = atom::simd_basics<T> &&
    (simd_with<T, typename T::value_type, A> ||
        mask_with<T, simd_lane_type_t<T>, A>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

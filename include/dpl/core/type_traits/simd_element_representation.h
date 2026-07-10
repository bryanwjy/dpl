// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/details/type_traits.h"
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/core/type_interface/enable_simd_abi.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename A, typename E>
struct simd_element_representation {};

DPL_EXPORT template <typename A, typename E>
using simd_element_representation_t =
    typename simd_element_representation<A, E>::type;

DPL_EXPORT template <typename A, integral E>
requires enable_simd_abi<A> && different_from<E, bool>
struct simd_element_representation<A, E> {
    using type = E;
};

DPL_EXPORT template <typename A, floating_point_like E>
requires enable_simd_abi<A>
struct simd_element_representation<A, E> {
    using type = E;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

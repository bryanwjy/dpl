// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_abi.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_like.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/integral.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename A, typename E>
struct simd_element_representation {};

template <typename A, typename E>
using simd_element_representation_t =
    typename simd_element_representation<A, E>::type;

template <typename A, integral E>
requires enable_simd_abi<A> && different_from<E, bool>
struct simd_element_representation<A, E> {
    using type = E;
};

template <typename A, floating_point_like E>
requires enable_simd_abi<A>
struct simd_element_representation<A, E> {
    using type = E;
};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

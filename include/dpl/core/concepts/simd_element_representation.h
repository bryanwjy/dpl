// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/basic_element.h"
#include "dpl/core/concepts/simd_abi.h"
#if !DPL_MODULES
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename, typename>
struct simd_element_representation {};

DPL_EXPORT template <typename A, typename E>
using simd_element_representation_t =
    typename simd_element_representation<A, E>::type;

DPL_EXPORT template <simd_abi A, basic_element E>
struct simd_element_representation<A, E> {
    using type = E;
};

DPL_EXPORT template <simd_abi A, enumeration E>
struct simd_element_representation<A, E> :
    simd_element_representation<A, underlying_type_t<E>> {};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

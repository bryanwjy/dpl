// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/simd_element_representation.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

template <typename E>
concept basic_element = !same_as<E, bool> && (integral<E> || floating_point<E>);

DPL_EXPORT template <typename E, typename A>
concept simd_element_for = simd_abi<A> && requires {
    // required by ABI to determine what vector type to use
    typename simd_element_representation_t<A, E>;
    requires basic_element<simd_element_representation_t<A, E>> &&
        sizeof(E) == sizeof(simd_element_representation_t<A, E>);
    typename A::template native_vector<E>;
    typename A::template native_mask<E>;
    typename A::template native_vector<simd_element_representation_t<A, E>>;
    typename A::template native_mask<simd_element_representation_t<A, E>>;
    requires same_as<
        typename A::template native_vector<simd_element_representation_t<A, E>>,
        typename A::template native_vector<E>>;
    requires same_as<
        typename A::template native_mask<simd_element_representation_t<A, E>>,
        typename A::template native_mask<E>>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename T, typename U, typename A = typename U::abi_type>
concept extended_vector_bit = simd_vector<T> &&
    common_size_with<simd_element_type_t<U>, simd_element_type_t<T>> &&
    unsigned_integral<simd_element_type_t<T>> &&
    common_abi_with<A, simd_abi_type_t<T>>;

template <typename T, typename U, typename A = typename U::abi_type>
concept canonical_vector_bit =
    extended_vector_bit<T, U, A> && same_as<A, simd_abi_type_t<T>>;

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_element_representation.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
inline constexpr bool is_canonical_type_v = false;
template <typename T>
inline constexpr bool is_canonical_type_v<T const> = is_canonical_type_v<T>;
template <typename T>
inline constexpr bool is_canonical_type_v<T volatile> = is_canonical_type_v<T>;
template <typename T>
inline constexpr bool is_canonical_type_v<T const volatile> =
    is_canonical_type_v<T>;

template <typename E, typename A>
inline constexpr bool is_canonical_type_v<basic_vector<E, A>> =
    is_same_v<simd_element_representation_t<A, E>, E>;
template <typename E, typename A>
inline constexpr bool is_canonical_type_v<basic_mask<E, A>> =
    is_same_v<simd_element_representation_t<A, E>, E>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

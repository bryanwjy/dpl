// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_vector.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename E, typename A>
concept simd_floating_point_for = floating_point<E> && simd_element_for<E, A>;
template <typename T>
concept floating_point_simd =
    simd_vector<T> && floating_point<simd_element_type_t<T>>;
template <typename T, typename A>
concept floating_point_simd_type_with_abi =
    simd_type_with_abi<T, A> && floating_point_simd<T>;

} // namespace datapar::internal

namespace datapar::fmath {
using __DPL datapar::internal::floating_point_simd;
using __DPL datapar::internal::floating_point_simd_type_with_abi;
using __DPL datapar::internal::simd_floating_point_for;

} // namespace datapar::fmath

DPL_DEFAULT_NAMESPACE_END

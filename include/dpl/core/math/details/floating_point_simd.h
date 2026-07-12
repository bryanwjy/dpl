// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/numbers/floating_point_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::fmath {
template <typename E, typename A>
concept simd_floating_point_for =
    floating_point_like<E> && simd_element_for<E, A>;
template <typename T>
concept floating_point_simd =
    simd_vector<T> && floating_point_like<simd_element_type_t<T>>;
template <typename T, typename A>
concept floating_point_simd_type_with_abi =
    internal::simd_type_with_abi<T, A> && floating_point_simd<T>;
} // namespace datapar::fmath

__DPL_DEFAULT_NAMESPACE_END

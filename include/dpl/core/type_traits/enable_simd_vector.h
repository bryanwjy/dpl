// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/simd_base.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/derived_from.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_vector_base : internal::simd_base {
    __DPL_HIDE_FROM_ABI constexpr ~simd_vector_base() = default;
};

template <typename T>
inline constexpr bool enable_simd_vector =
    (derived_from<T, simd_vector_base<T>> &&
        derived_from<T, internal::simd_base>);
template <typename E, typename A>
inline constexpr bool enable_simd_vector<basic_vector<E, A>> = true;

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END

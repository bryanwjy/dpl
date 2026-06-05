// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/derived_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_vector_base {
    __DPL_HIDE_FROM_ABI constexpr ~simd_vector_base() = default;
};

DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_vector = derived_from<T, simd_vector_base<T>>;
DPL_EXPORT template <typename E, typename A>
inline constexpr bool enable_simd_vector<basic_vector<E, A>> = true;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULE
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/is_constructible.h"
#  include "dpl/std/type_traits/is_convertible.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_mask = false;
DPL_EXPORT template <simd_element T, simd_abi Abi>
inline constexpr bool enable_simd_mask<simd_mask<T, Abi>> = true;

DPL_EXPORT template <typename M>
concept simd_mask_type = enable_simd_mask<M> && requires(M const mask) {
    typename M::value_type;
    typename M::abi_type;
    typename M::simd_type;
    requires same_as<typename M::value_type, bool> &&
        simd_abi<typename M::abi_type> && simd_type<typename M::simd_type>;
    {
        +mask
    } -> same_as<typename M::abi_type::template native_mask<
        typename M::simd_type::value_type>>;
};

DPL_EXPORT template <simd_mask_type T>
struct simd_element_type<T> : simd_element_type<typename T::simd_type> {};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

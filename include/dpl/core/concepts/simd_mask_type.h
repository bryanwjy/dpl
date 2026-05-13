// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element_for.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_mask = false;
DPL_EXPORT template <simd_abi A, simd_element_for<A> E>
inline constexpr bool enable_simd_mask<basic_simd_mask<E, A>> = true;

namespace atom {
template <typename M>
concept simd_mask_type = enable_simd_mask<M> && requires(M const mask) {
    typename M::simd_type;
    requires same_as<typename M::value_type, bool> &&
        simd_type<typename M::simd_type>;
    requires explicitly_convertible_to<M,
        typename M::abi_type::template native_mask<
            typename M::simd_type::value_type>>;
};
} // namespace atom

DPL_EXPORT template <typename M>
concept simd_mask_type = atom::simd_basics<M> && atom::simd_mask_type<M>;
DPL_EXPORT template <typename T>
concept scalable_mask = simd_mask_type<T> && scalable_abi<typename T::abi_type>;
DPL_EXPORT template <typename T>
concept fixed_width_mask =
    simd_mask_type<T> && fixed_width_abi<typename T::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

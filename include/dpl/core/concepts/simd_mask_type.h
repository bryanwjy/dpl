// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/semiregular.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_mask = false;
DPL_EXPORT template <simd_element T, simd_abi Abi>
inline constexpr bool enable_simd_mask<basic_simd_mask<T, Abi>> = true;

DPL_EXPORT template <typename M>
concept simd_mask_type = enable_simd_mask<M> && is_object_v<M> &&
    semiregular<M> && is_trivially_copyable_v<M> && requires(M const mask) {
        typename M::value_type;
        typename M::abi_type;
        typename M::simd_type;
        requires same_as<typename M::value_type, bool> &&
            simd_abi<typename M::abi_type> && simd_type<typename M::simd_type>;
        requires explicitly_convertible_to<M,
            typename M::abi_type::template native_mask<
                typename M::simd_type::value_type>>;
    };

DPL_EXPORT template <simd_mask_type T>
struct simd_element_type<T> : simd_element_type<typename T::simd_type> {};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END

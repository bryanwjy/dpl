// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/immediate_mask.h" // IWYU pragma: export
#include "dpl/xmm/concepts.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

template <xmm_element T>
using bit_type_for = bit_type_t<element_count<T>>;

template <xmm_element T, bit_type_for<T> V>
using mask_for = basic_mask<element_count<T>, V>;

template <xmm_element T, bit_type_for<T> V>
inline constexpr mask_for<T, V> mask_for_v{};

template <xmm_element T>
inline consteval auto mask(all_t<T> tag) noexcept {
    return basic_mask{xmm_abi{}, tag};
}

template <xmm_element T>
inline consteval auto mask(zero_t<T> tag) noexcept {
    return basic_mask{xmm_abi{}, tag};
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END

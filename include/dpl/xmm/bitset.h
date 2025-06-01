// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/fwd.h"

#include "dpl/bitset.h" // IWYU pragma: export
#include "dpl/xmm/blob.h"
#include "dpl/xmm/concepts.h"
#include "dpl/xmm/constants.h"

#if !DPL_MODULES
#  include <climits>
#  include <type_traits>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

template <xmm_element T>
using bitset_for = bitset<element_count<T>>;

template <xmm_element T>
consteval bitset<xmm_abi::size> operator*(
    std::type_identity_t<bitset_for<T>> set, lsb_t<T>) noexcept {
    return set.template zero_extend<xmm_abi::size / element_count<T>>();
}

template <xmm_element T>
consteval bitset<xmm_abi::size> operator*(
    std::type_identity_t<bitset_for<T>> set, msb_t<T>) noexcept {
    return set * lsb<T> << (sizeof(T) * CHAR_BIT - 1);
}

template <xmm_element T>
consteval bitset<xmm_abi::size> operator*(
    std::type_identity_t<bitset_for<T>> set, all_t<T>) noexcept {
    return set.template repeat<xmm_abi::size / element_count<T>>();
}

template <xmm_element T>
consteval bitset<xmm_abi::size> operator*(
    lsb_t<T> tag, std::type_identity_t<bitset_for<T>> set) noexcept {
    return set * tag;
}

template <xmm_element T>
consteval bitset<xmm_abi::size> operator*(
    msb_t<T> tag, std::type_identity_t<bitset_for<T>> set) noexcept {
    return set * tag;
}

template <xmm_element T>
consteval bitset<xmm_abi::size> operator*(
    all_t<T> tag, std::type_identity_t<bitset_for<T>> set) noexcept {
    return set * tag;
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END

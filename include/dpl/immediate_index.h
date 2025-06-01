// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/bitset.h"
#include "dpl/concepts.h"
#include "dpl/constants.h"
#include "dpl/native.h"
#include "dpl/numeric.h"

#if !DPL_MODULES
#  include <bit>
#  include <utility>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT_BEGIN

template <size_t... Is>
struct basic_indices;

template <size_t... Is>
requires (std::has_single_bit(sizeof...(Is)))
struct basic_indices {
    static_assert((... && (Is < sizeof...(Is))), "invalid index");

    __DPL_HIDE_FROM_ABI consteval basic_indices() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr basic_indices(
        std::index_sequence<Is...>) noexcept {}

    template <abi_tag Abi, simd_element T>
    requires std::constructible_from<basic_indices,
        std::make_index_sequence<Abi::size>>
    __DPL_HIDE_FROM_ABI constexpr basic_indices(Abi, iota_t<T>) noexcept {}
};

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END

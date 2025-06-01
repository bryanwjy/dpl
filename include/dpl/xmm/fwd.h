// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/fwd.h"

DPL_EXPORTED_DEFAULT_NAMESPACE_BEGIN

struct xmm_abi {
    static constexpr size_t size = 16;
    static constexpr size_t alignment = 16;
};

template <>
inline constexpr bool is_abi_tag_v<xmm_abi> = true;

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

// IWYU pragma: begin_exports
template <typename>
class __DPL_PUBLIC_TEMPLATE vector;

template <typename>
class __DPL_PUBLIC_TEMPLATE scalar;

template <typename>
class __DPL_PUBLIC_TEMPLATE vector_mask;

template <typename>
class __DPL_PUBLIC_TEMPLATE scalar_mask;

template <size_t W, bit_type_t<W> Value>
struct basic_mask;

// IWYU pragma: end_exports

template <typename T>
struct __DPL_PUBLIC_TEMPLATE native_type {};

template <typename T>
using native_type_t = typename native_type<T>::type;

template <typename>
__DPL_HIDE_FROM_ABI void reinterpret_as(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void shuffle(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void permute(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void select(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void negate(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void negate(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void hmin(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void hmin(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void hmax(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void hmax(...) noexcept = delete;
template <auto...>
__DPL_HIDE_FROM_ABI void reduce(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void reduce(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void min(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void max(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void abs(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void rsqrt(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void rcp(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void sqrt(...) noexcept = delete;

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_EXPORTED_DEFAULT_NAMESPACE_END

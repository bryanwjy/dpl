// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/fwd.h"

#include "dpl/concepts.h" // IWYU pragma: export

#if !DPL_MODULES
#  include <concepts>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept xmm_element = simd_element<T> && sizeof(T) <= xmm::register_size;

template <typename T>
concept xmm_scalar_element =
    scalar_element_type<T> && sizeof(T) <= xmm::register_size;

template <typename T>
concept xmm_type = simd_type<T> && xmm_element<typename T::element_type> &&
    std::same_as<xmm::native_type_t<typename T::element_type>,
        typename T::native_type>;

template <typename T, typename E>
concept xmm_type_for = xmm_type<T> && simd_type_for<T, E> &&
    std::same_as<xmm::native_type_t<E>, typename T::native_type>;

template <simd_element T>
inline constexpr auto is_simd_type_v<xmm::vector<T>> = true;

template <xmm_scalar_element T>
inline constexpr auto is_simd_type_v<xmm::scalar<T>> = true;

template <simd_element T>
inline constexpr auto is_simd_mask_type_v<xmm::vector_mask<T>> = true;

template <xmm_scalar_element T>
inline constexpr auto is_simd_mask_type_v<xmm::scalar_mask<T>> = true;

template <xmm_element T, xmm_element E>
struct __DPL_PUBLIC_TEMPLATE rebind_simd<xmm::vector_mask<T>> {
    using type = xmm::vector_mask<E>;
};

template <xmm_element T, xmm_element E>
struct __DPL_PUBLIC_TEMPLATE rebind_simd<xmm::vector<T>> {
    using type = xmm::vector<E>;
};

template <xmm_scalar_element T, xmm_scalar_element E>
struct __DPL_PUBLIC_TEMPLATE rebind_simd<xmm::scalar_mask<T>> {
    using type = xmm::scalar_mask<E>;
};

template <xmm_scalar_element T, xmm_scalar_element E>
struct __DPL_PUBLIC_TEMPLATE rebind_simd<xmm::scalar<T>> {
    using type = xmm::scalar<E>;
};

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

template <xmm_element T>
inline constexpr size_t element_count = xmm_abi::size / sizeof(T);

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_DEFAULT_NAMESPACE_END

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT_BEGIN

template <typename T>
inline constexpr bool is_simd_type_v = false;

template <typename T>
inline constexpr bool is_simd_mask_type_v = false;

template <typename T>
inline constexpr bool is_abi_tag_v = false;

template <typename T, typename E>
struct __DPL_PUBLIC_TEMPLATE rebind_simd {};

template <typename T, typename E>
using rebind_simd_t = typename rebind_simd<T>::type;

template <size_t Width>
struct __DPL_PUBLIC_TEMPLATE bit_type {};

template <size_t Width>
using bit_type_t = typename bit_type<Width>::type;

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/arithmetic_type.h"
#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_bits_with.h"
#include "dpl/core/concepts/common_float_with.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * Arithmetic operations (excluding division for integers) may use the same
 * instruction, signed-ness do not matter. Enums are only allowed if arithmetic
 * enabled.
 */
DPL_EXPORT template <typename A, typename B>
concept common_arithmetic_with =
    common_bits_with<A, B> && arithmetic_type<A> && arithmetic_type<B> &&
    (common_float_with<A, B> || (integral<A> && integral<B>));

DPL_EXPORT template <typename A, typename B>
concept simd_common_arithmetic_with = simd_common_abi_with<A, B> &&
    common_arithmetic_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#include "dpl/core/math/details/compliance.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/exponent_bias.h"
#  include "dpl/core/immediate/constants/exponent_bits.h"
#  include "dpl/core/immediate/constants/mantissa_width.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/bit_cast.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
template <canonical_vector T>
requires binary_layout_floating_point<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL ilogb(compliance::unsafe_t, T val) noexcept {
    // disregards subnormal/inf/nans
    using E = simd_element_type_t<T>;
    constexpr auto exp_bits =
        __DPL bit_cast<E>(floating_point_traits<E>::exponent_mask);
    constexpr auto shift =
        __DPL countr_zero(floating_point_traits<E>::exponent_mask);
    auto const bexp = dx::bwshift_right(dx::bwand(val, exp_bits), imm<shift>);

    using sint_t = dx::signed_representation_t<T>;
    return dx::subtract(
        dx::reinterpret<sint_t>(bexp), floating_point_traits<E>::exponent_bias);
}
} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END

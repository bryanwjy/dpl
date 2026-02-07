// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/compliance.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/digits.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/exponent_bits.h"
#  include "dpl/core/type_traits/to_integral.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <floating_point T, simd_abi A>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr auto DPL_VECTORCALL
    ilogb(compliance::unsafe_t, basic_simd<T, A> val) noexcept {
    // disregards subnormal/inf/nans
    auto const bexp = (val & dx::exponent_bits) >> imm<dx::digits<T>>;
    using int_type = dx::to_signed_integral_t<T>;
    return dx::reinterpret<int_type>(bexp) -
        dx::broadcast<int_type, A>(dx::exponent_bias<T>);
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END

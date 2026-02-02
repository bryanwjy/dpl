// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/accuracy.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <simd_abi A, accuracy_tag Tag>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
static constexpr basic_simd<float, A> DPL_VECTORCALL
    rsqrt(Tag, basic_simd<float, A> val) noexcept {
    using simd = basic_simd<float, A>;
    auto const x2 = val * 0.5f;
    auto const i = dx::sub(0x5f3759df, (dx::reinterpret<int32>(val) >> imm<1>));
    constexpr auto newton = [](basic_simd<float, A> val,
                                basic_simd<float, A> x2) noexcept {
        auto const sq = dx::mul(val, val);
        constexpr float threehalves = 1.5f;
        return val * dx::fnmadd(sq, x2, threehalves);
    };

    val = newton(dx::reinterpret<simd>(i), x2);
    if constexpr (same_as<Tag, accuracy::maximum_t>) {
        val = newton(val, x2);
    }

    return val;
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END

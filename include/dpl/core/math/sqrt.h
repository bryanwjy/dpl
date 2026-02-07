// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/accuracy.h"
#include "dpl/core/math/internal/decompose.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/rsqrt2.h"
#include "dpl/core/math/isfinite.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

void sqrt(...) noexcept = delete;

struct sqrt_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        // 2 * sqrt(2)
        constexpr auto two = dx::broadcast<E, A>(2.0);
        constexpr auto vsqrt8 =
            dx::broadcast<E, A>(2.8284271247461900976033774484);

        auto const decomp = fmath::frexp(val);
        auto const remtwo = decomp.exponent & dx::one;
        auto const ifodd = dx::select(remtwo == dx::zero, vsqrt8, two);
        auto const sig = (ifodd * decomp.significand) *
            mx::rsqrt2(mx::accuracy::maximum, decomp.significand);
        auto const result = mx::ldexp(mx::compliance::unsafe, //
            sig, (decomp.exponent - dx::one) >> imm<1>);

        return dx::select(dx::isfinite(val) && val != dx::zero,
            dx::bit_fill(val < dx::zero, result), val);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires { sqrt(internal::abi<T>, val); }) {
            if not consteval {
                return sqrt(internal::abi<T>, val);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept
        -> equivalent_simd_as<T> auto {
        if constexpr (requires(T val) { sqrt(internal::abi<T>, val); }) {
            return sqrt(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sqrt_t sqrt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

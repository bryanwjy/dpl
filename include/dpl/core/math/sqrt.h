// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/accuracy.h"
#include "dpl/core/math/internal/decompose.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/rsqrt.h"
#include "dpl/core/math/isfinite.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
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
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        auto const decomp = mx::decompose(val);
        auto const sig = dx::mul(decomp.significand,
            mx::rsqrt(mx::accuracy::maximum, decomp.significand));
        auto const result =
            mx::ldexp(mx::compliance::unsafe, sig, decomp.exponent >> imm<1>);
        return dx::select(dx::isfinite(val) & (val != dx::zero),
            dx::bit_fill(dx::cmplt(val, dx::zero), result), val);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept {
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

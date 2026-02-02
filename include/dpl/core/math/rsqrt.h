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
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;
struct rsqrt_t {
private:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        auto const decomp = mx::decompose(val);
        auto const result = mx::ldexp(mx::compliance::unsafe,
            mx::rsqrt(mx::accuracy::speed, decomp.significand),
            -(decomp.exponent >> imm<1>));
        return dx::select(
            dx::isfinite(val), dx::bit_fill(val <= dx::zero, result), val);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        if constexpr (requires { rsqrt(internal::abi<T>, val); }) {
            if not consteval {
                return rsqrt(internal::abi<T>, val);
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
        if constexpr (requires(T val) { rsqrt(internal::abi<T>, val); }) {
            return rsqrt(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END

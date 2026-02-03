// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/decompose.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/isinf.h"
#include "dpl/core/math/isnan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

void rcp(...) noexcept = delete;

struct rcp_t {
private:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto newton_step(
        basic_simd<float, A> y, basic_simd<float, A> x) noexcept {
        using simd = basic_simd<float, A>;
        return y * dx::fnmadd(x, y, 2.0f);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<float, A> calculate(
        basic_simd<float, A> val) noexcept {
        using simd = basic_simd<float, A>;
        auto const i = dx::sub(0x7EF311C3u, dx::reinterpret<uint32>(val));
        auto result = dx::reinterpret<float>(i);
        result = newton_step(result, val);
        return result;
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<float, A> val) noexcept {
        auto const decomp = mx::decompose(val);
        auto const sig = calculate(decomp.significand);
        auto const result =
            mx::ldexp(mx::compliance::unsafe, sig, -decomp.exponent);
        return dx::select(
            dx::isnan(val), val, dx::bit_drop(dx::isinf(val), result));
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        if constexpr (requires { rcp(internal::abi<T>, val); }) {
            if not consteval {
                return rcp(internal::abi<T>, val);
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
        if constexpr (requires(T val) { rcp(internal::abi<T>, val); }) {
            return rcp(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END

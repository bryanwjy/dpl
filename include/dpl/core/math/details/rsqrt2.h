// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#include "dpl/core/math/details/accuracy.h" // IWYU pragma: export
#include "dpl/core/math/details/floating_point_simd.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/mulx.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

/**
 * Approximates 1 / sqrt( 2x ) where x is in the interval [0.5,1)
 */
struct rsqrt2_t {
private:
    static constexpr fmath::polynomial<1.0, //
        -0.49718099016606776,               //
        0.34347649576578587,                //
        -0.1929776112466749,                //
        0.05385400159427293>
        polynomial;

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T refine(T poly, T, immediate<0>) noexcept {
        return poly;
    }

    template <canonical_vector T, int N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL refine(
        T poly, T half_x, immediate<N>) noexcept {
        static_assert(N >= 1);
        constexpr simd_element_type_t<T> threehalves = 1.5;
        auto result = dx::multiply(poly, poly);
        result = dx::nmuladd(result, half_x, threehalves);
        result = dx::multiply(result, poly);
        return refine(result, half_x, imm<N - 1>);
    }

    struct one : broadcastable_base<one> {
        __DPL_HIDE_FROM_ABI explicit constexpr one() noexcept = default;

        template <floating_point_like T>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
        constexpr operator T(this one_t) noexcept {
            return 0.7071067811865475244008443;
        }
    };

public:
    template <canonical_vector T, accuracy_tag Tag>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(Tag, T val) noexcept {
        constexpr simd_element_type_t<T> two = 2.0;
        auto const vtwo = dx::broadcast<T>(two);
        auto const vone = dx::broadcast<T>(dx::one);
        auto poly = polynomial(dx::mulsub(vtwo, val, vone));
        if constexpr (is_same_v<accuracy::speed_t, Tag>) {
            constexpr auto steps =
                digits_v<simd_element_type_t<T>> / digits_v<double>;
            return refine(poly, val, imm<steps>);
        } else {
            constexpr auto steps =
                digits_v<simd_element_type_t<T>> / digits_v<float>;
            return refine(poly, val, imm<steps>);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(accuracy_tag auto, dx::one_t) noexcept {
        return rsqrt2_t::one{};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(dx::one_t) noexcept {
        return rsqrt2_t::one{};
    }
};

inline constexpr rsqrt2_t rsqrt2{};

} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END

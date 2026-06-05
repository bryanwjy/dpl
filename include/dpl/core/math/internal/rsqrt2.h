// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/fwd.h" // IWYU pragma: export

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/accuracy.h" // IWYU pragma: export
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/polynomial.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
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

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto refine(basic_vector<E, A> poly,
        basic_vector<E, A> half_x, immediate<0>) noexcept {
        return poly;
    }

    template <floating_point E, simd_abi A, int N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL refine(basic_vector<E, A> poly,
        basic_vector<E, A> half_x, immediate<N>) noexcept {
        static_assert(N >= 1);
        constexpr E threehalves = 1.5;
        auto const result = poly * dx::fnmadd(poly * poly, half_x, threehalves);
        return refine(result, half_x, imm<N - 1>);
    }

    struct one : broadcastable_base<one> {
        __DPL_HIDE_FROM_ABI explicit constexpr one() noexcept = default;

        template <floating_point T>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
        constexpr operator T(this one_t) noexcept {
            return 0.7071067811865475244008443;
        }
    };

public:
    template <simd_abi A, simd_floating_point_for<A> E, accuracy_tag Tag>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(Tag, basic_vector<E, A> val) noexcept {
        constexpr E two = 2.0;
        auto poly = polynomial(dx::fmsub(dx::broadcast<A>(two), val, dx::one));
        if constexpr (same_as<accuracy::speed_t, Tag>) {
            return refine(poly, val, imm<digits_v<E> / digits_v<double>>);
        } else {
            return refine(poly, val, imm<digits_v<E> / digits_v<float>>);
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

inline constexpr fmath::rsqrt2_t rsqrt2{};

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END

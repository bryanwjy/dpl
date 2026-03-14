// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd_with_abi.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/exponent_mask.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/bit/popcount.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void ldexp(...) noexcept = delete;

struct ldexp_t : binary_operation_base<ldexp_t> {
    friend binary_operation_base<ldexp_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        {
            ldexp(internal::abi<A>, left, right)
        } -> floating_point_simd_with_abi<A>;
    }
    {
        return ldexp(internal::abi<A>, left, right);
    }

    template <signed_integral E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<E, A> signbit(basic_simd<E, A> val) noexcept {
        constexpr auto shift = sizeof(E) * char_bit_v - 1;
        return val >> imm<shift>;
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL fallback(
        basic_simd<E, A> num, basic_simd<signed_rep_t<E>, A> exp) noexcept {

        constexpr auto mantissa_shift = imm<dx::mantissa_width_v<E>>;
        using sint = signed_rep_t<E>;
        constexpr auto exp_mask =
            dx::exponent_mask_v<E, sint> >> mantissa_shift;
        constexpr auto exp_bias = dx::exponent_bias<E>;
        constexpr auto chunk =
            __DPL popcount(static_cast<unsigned>(exp_bias)) - 1;
        constexpr auto lshift = imm<chunk>;
        constexpr auto rshift = imm<chunk - 2>;
        auto const sign = signbit(exp);
        auto m = (((sign + exp) >> lshift) - sign) << rshift;
        exp = exp - (m << imm<2>);

        m += exp_bias;
        m = dx::bit_drop(dx::zero > m, m);
        m = dx::select(m > exp_mask, exp_mask, m);

        using simdi = basic_simd<sint, A>;
        auto u = dx::reinterpret<E>(m << mantissa_shift);

        num *= [](auto u2) { return u2 * u2; }(u * u);

        return num * dx::reinterpret<E>((exp + exp_bias) << mantissa_shift);
    }

    template <simd_type T>
    using int_simd DPL_NODEBUG =
        rebind_simd_t<T, signed_rep_t<typename T::value_type>>;

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T num, int_simd<T> exp) noexcept {
        if constexpr (requires {
                          {
                              ldexp(internal::abi<T>, num, exp)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(num, exp);
                } else {
                    return ldexp(internal::abi<T>, num, exp);
                }
            } else {
                return ldexp(internal::abi<T>, num, exp);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(num, exp);
        } else {
            return operator()(dx::to_basic_type(num), dx::to_basic_type(exp));
        }
    }

    using binary_operation_base<ldexp_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::ldexp_t ldexp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

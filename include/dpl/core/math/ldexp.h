// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/exponent_mask.h"
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

template <typename A, typename L, typename R>
concept unqualified_ldexp =
    requires(L lhs, R rhs) { ldexp(internal::abi<A>, lhs, rhs); };

struct ldexp_t : binary_operation_base<ldexp_t> {
    friend binary_operation_base<ldexp_t>;

    template <simd_abi A, typename L, typename R>
    requires unqualified_ldexp<A, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
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
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> num,
            basic_simd<to_signed_integral_t<E>, A> exp) noexcept {

        using sint = to_signed_integral_t<E>;
        constexpr auto exp_mask = dx::exponent_mask_v<E, sint> >> dx::digits<E>;
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
        auto u = dx::reinterpret<E>(m << dx::digits<E>);

        num *= [](auto u2) { return u2 * u2; }(u * u);

        return num * dx::reinterpret<E>((exp + exp_bias) << dx::digits<E>);
    }

    template <simd_type T>
    using int_simd DPL_NODEBUG =
        rebind_simd_t<T, to_signed_integral_t<typename T::value_type>>;

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T num, int_simd<T> exp) noexcept {
        if constexpr (unqualified_ldexp<T, T, int_simd<T>>) {
            if consteval {
                return fallback(num, exp);
            } else {
                return ldexp(internal::abi<T>, num, exp);
            }
        } else {
            return fallback(num, exp);
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T num, int_simd<T> exp) noexcept
        -> equivalent_simd_as<T> auto {
        if constexpr (unqualified_ldexp<T, T, int_simd<T>>) {
            return ldexp(internal::abi<T>, num, exp);
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

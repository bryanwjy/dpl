// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/integral_simd.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/exponent_mask.h"
#  include "dpl/core/constants/mantissa_width.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/std/bit/popcount.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept ldexp_integral =
    simd_vector<T> && signed_integral<typename T::value_type>;

void ldexp(...) noexcept = delete;

struct ldexp_t;

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_canonical_ldexp = requires(T val, I exp) {
    {
        ldexp(internal::abi<T>, val, exp)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_extended_ldexp = requires(T val, I exp) {
    { ldexp(val, exp) } -> extended_operation_vector<A>;
};

template <typename T, typename I>
concept expression_ldexp = (simd_expression<T> || simd_expression<I>) &&
    invocable<ldexp_t, simd_expression_result_t<T>,
        simd_expression_result_t<I>>;

template <typename T, typename I>
concept decayable_ldexp =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_vector_for<I, operation_category::lane_agnostic> &&
    regular_invocable<ldexp_t, canonical_type_t<T>, canonical_type_t<I>>;

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept extended_ldexp = unqualified_extended_ldexp<T, I, A> ||
    expression_ldexp<T, I> || decayable_ldexp<T, I>;

struct ldexp_t :
    private binary_operation_base<ldexp_t>,
    private mx::masked_operation<ldexp_t> {
    friend binary_operation_base<ldexp_t>;
    friend mx::masked_operation<ldexp_t>;

    template <simd_abi A, typename L, typename R>
    requires (canonical_vector<L> || canonical_vector<R>) &&
        unqualified_canonical_ldexp<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return ldexp(internal::abi<A>, left, right);
    }

    template <simd_abi A, typename L, typename R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_ldexp<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return ldexp(left, right);
    }

    template <signed_integral E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> signbit(
        basic_vector<E, A> val) noexcept {
        constexpr auto shift = sizeof(E) * char_bit_v - 1;
        return val >> imm<shift>;
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> num,
            basic_vector<signed_representation_t<E>, A> exp) noexcept {

        constexpr auto mantissa_shift = imm<dx::mantissa_width_v<E>>;
        using sint = signed_representation_t<E>;
        constexpr auto exp_mask =
            dx::exponent_mask_v<E, sint> >> mantissa_shift;
        constexpr auto exp_bias = dx::exponent_bias<E>;
        constexpr auto chunk =
            __DPL popcount(static_cast<unsigned>(exp_bias)) - 1;
        constexpr auto lshift = imm<chunk>;
        constexpr auto rshift = imm<chunk - 2>;
        auto const sign = ldexp_t::signbit(exp);
        auto m = (((sign + exp) >> lshift) - sign) << rshift;
        exp = exp - (m << imm<2>);

        m += exp_bias;
        m = dx::max(dx::zero, m);
        m = dx::min(exp_mask, m);

        using simdi = basic_vector<sint, A>;
        auto u = dx::reinterpret<E>(m << mantissa_shift);

        num *= [](auto u2) { return u2 * u2; }(u * u);

        return num * dx::reinterpret<E>((exp + exp_bias) << mantissa_shift);
    }

    template <simd_vector T>
    using int_simd DPL_NODEBUG =
        rebind_simd_t<T, signed_representation_t<typename T::value_type>>;

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires mx::canonical_masked_math_operator<ldexp_t, S, M, L, R> &&
        requires(S src, M mask, L lhs, R rhs) {
            ldexp(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires mx::extended_masked_math_operator<ldexp_t, S, M, L, R> &&
        requires(S src, M mask, L lhs, R rhs) { ldexp(src, mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, L lhs, R rhs) noexcept {
        return ldexp(src, mask, lhs, rhs);
    }

    template <typename M, simd_vector L, simd_vector R>
    requires mx::canonical_masked_math_zoperator<ldexp_t, M, L, R> &&
        requires(M mask, L lhs, R rhs) {
            ldexp(internal::abi<common_abi_t<L, R>>, dx::zero, mask, lhs, rhs);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, L lhs, R rhs) noexcept {
        return ldexp(
            internal::abi<common_abi_t<L, R>>, dx::zero, mask, lhs, rhs);
    }

    template <typename M, simd_vector L, simd_vector R>
    requires mx::extended_masked_math_zoperator<ldexp_t, M, L, R> &&
        requires(M mask, L lhs, R rhs) { ldexp(dx::zero, mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, L lhs, R rhs) noexcept {
        return ldexp(dx::zero, mask, lhs, rhs);
    }

public:
    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires signed_integral<I> && floating_point<E> && common_size_with<E, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> num, basic_vector<I, A> exp) noexcept {
        if constexpr (unqualified_canonical_ldexp<basic_vector<E, A>,
                          basic_vector<I, A>, A>) {
            if consteval {
                return ldexp_t::fallback(num, exp);
            } else {
                return ldexp(internal::abi<A>, num, exp);
            }
        } else {
            return ldexp_t::fallback(num, exp);
        }
    }

    template <simd_abi LA, simd_element_for<LA> E, simd_abi RA,
        simd_element_for<RA> I>
    requires signed_integral<I> && common_size_with<E, I> &&
        (!floating_point<E> || different_from<LA, RA>) &&
        unqualified_canonical_ldexp<basic_vector<E, LA>, basic_vector<I, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> num, basic_vector<I, RA> exp) noexcept {
        using A = common_abi_t<LA, RA>;
        return ldexp(internal::abi<A>, num, exp);
    }

    template <simd_vector T, simd_vector I>
    requires signed_integral<typename I::value_type> &&
        common_size_with<typename T::value_type, typename I::value_type> &&
        (extended_vector<T> || extended_vector<I>) && extended_ldexp<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T num, I exp) noexcept {
        if constexpr (unqualified_extended_ldexp<T, I>) {
            return ldexp(num, exp);
        } else if constexpr (expression_ldexp<T, I>) {
            return operator()(dx::evaluate(num), dx::evaluate(exp));
        } else {
            return operator()(dx::to_canonical(num), dx::to_canonical(exp));
        }
    }

    using binary_operation_base<ldexp_t>::operator();
    using mx::masked_operation<ldexp_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::ldexp_t ldexp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

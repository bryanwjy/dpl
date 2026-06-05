// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/accuracy.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/internal/rsqrt2.h"
#include "dpl/core/math/isfinite.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void sqrt(...) noexcept = delete;

struct sqrt_t;

template <typename T>
concept unqualified_canonical_sqrt = requires(T val) {
    {
        sqrt(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_sqrt = requires(T val) {
    { sqrt(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_sqrt =
    simd_expression<T> && invocable<sqrt_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_sqrt =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<sqrt_t, canonical_type_t<T>>;

template <typename T>
concept extended_sqrt =
    unqualified_extended_sqrt<T> || expression_sqrt<T> || decayable_sqrt<T>;

struct sqrt_t : private mx::masked_operation<sqrt_t> {
private:
    friend mx::masked_operation<sqrt_t>;

    template <typename E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        // 2 * sqrt(2)
        constexpr auto two = dx::broadcast<E, A>(2.0);
        constexpr auto vsqrt8 =
            dx::broadcast<E, A>(2.8284271247461900976033774484);

        auto const decomp = dx::frexp(val);
        auto const remtwo = decomp.exp & dx::one;
        auto const ifodd = dx::select(remtwo == dx::zero, vsqrt8, two);
        auto const sig =
            (ifodd * decomp.fr) * mx::rsqrt2(mx::accuracy::maximum, decomp.fr);
        auto const result = mx::ldexp(mx::compliance::unsafe, //
            sig, (decomp.exp - dx::one) >> imm<1>);

        return dx::select(dx::isfinite(val) && val != dx::zero,
            dx::select(val < dx::zero, dx::all_bits, result), val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<sqrt_t, S, M, T> &&
        requires(
            S src, M mask, T val) { sqrt(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sqrt(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<sqrt_t, S, M, T> &&
        requires(S src, M mask, T val) { sqrt(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sqrt(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<sqrt_t, M, T> &&
        requires(M mask, T val) { sqrt(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sqrt(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<sqrt_t, M, T> &&
        requires(M mask, T val) { sqrt(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sqrt(dx::zero, mask, val);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_sqrt<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return sqrt(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_sqrt<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return sqrt(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_sqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_sqrt<T>) {
            return sqrt(val);
        } else if constexpr (expression_sqrt<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<sqrt_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sqrt_t sqrt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

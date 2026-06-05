// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/nan.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/type_traits/simd_traits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void log2(...) noexcept = delete;

struct log2_t;

template <typename T>
concept unqualified_canonical_log2 = requires(T val) {
    {
        log2(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_log2 = requires(T val) {
    { log2(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_log2 =
    simd_expression<T> && invocable<log2_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_log2 =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<log2_t, canonical_type_t<T>>;

template <typename T>
concept extended_log2 =
    unqualified_extended_log2<T> || expression_log2<T> || decayable_log2<T>;

struct log2_t : private mx::masked_operation<log2_t> {
private:
    friend mx::masked_operation<log2_t>;

    template <floating_point E, simd_abi A>
    requires (dx::digits_v<E> <= dx::digits_v<float>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        using simdf = basic_vector<E, A>;
        auto const decomp =
            dx::frexp(val, frexp_reduced | frexp_floating_point);
        constexpr auto n_one = fmath::single(dx::broadcast<A, E>(-1));
        constexpr auto one = fmath::single(dx::broadcast<A, E>(1));
        auto const x = (n_one + decomp.fr) / (one + decomp.fr);
        auto const x2 = x.upper * x.upper;
        constexpr fmath::polynomial<0.9618012905120f, //
            0.5764790177e+0f,                         //
            0.4374550283e+0f>
            polynomial;
        auto const t = polynomial(x2);
        constexpr auto inv_halfln2 = fmath::make_pair<E, A>(
            2.8853900432586669922f, 3.2734474483568488616e-08f);
        auto s = decomp.exp + x * inv_halfln2;
        s = s + x2 * x * t;
        auto result = s.upper + s.lower;

        return dx::fixup(val, result,
            fpfix::condition<fpfix::negative, dx::nan> |
                fpfix::condition<fpfix::zero, -dx::infinity> |
                fpfix::condition<fpfix::infinity, fpfix::revert> |
                fpfix::condition<fpfix::nan, fpfix::revert>);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<double, A> val) noexcept {
        auto const decomp =
            dx::frexp(val, frexp_reduced | frexp_floating_point);
        constexpr auto n_one = fmath::single(dx::broadcast<A, double>(-1));
        constexpr auto one = fmath::single(dx::broadcast<A, double>(1));
        auto const x = (n_one + decomp.fr) / (one + decomp.fr);
        auto const x2 = x.upper * x.upper;
        constexpr fmath::polynomial<0.96179669392608091449,
            0.5770780162997058982, 0.4121985945485324709, 0.3205977477944495502,
            0.2623708057488514656, 0.2200768693152277689, 0.2211941750456081490>
            polynomial;
        auto const t = polynomial(x2);
        constexpr auto inv_halfln2 = fmath::make_pair<double, A>(
            2.885390081777926774, 6.0561604995516736434e-18);

        auto s = decomp.exp + x * inv_halfln2;
        s = s + x2 * x * t;
        auto result = s.upper + s.lower;

        return dx::fixup(val, result,
            fpfix::condition<fpfix::negative, dx::nan> |
                fpfix::condition<fpfix::zero, -dx::infinity> |
                fpfix::condition<fpfix::infinity, fpfix::revert> |
                fpfix::condition<fpfix::nan, fpfix::revert>);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<log2_t, S, M, T> &&
        requires(
            S src, M mask, T val) { log2(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return log2(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<log2_t, S, M, T> &&
        requires(S src, M mask, T val) { log2(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return log2(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<log2_t, M, T> &&
        requires(M mask, T val) { log2(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return log2(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<log2_t, M, T> &&
        requires(M mask, T val) { log2(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return log2(dx::zero, mask, val);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_log2<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return log2(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_log2<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return log2(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_log2<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_log2<T>) {
            return log2(val);
        } else if constexpr (expression_log2<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
    using mx::masked_operation<log2_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::log2_t log2{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

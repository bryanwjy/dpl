// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/infinity.h"
#  include "dpl/core/constants/nan.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void log2(...) noexcept = delete;

template <typename T>
concept unqualified_log2 = floating_point_simd<T> && requires(T val) {
    { log2(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

struct log2_t {
private:
    template <floating_point E, simd_abi A>
    requires (dx::digits_v<E> <= dx::digits_v<float>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        using simdf = basic_simd<E, A>;
        auto const decomp =
            fmath::frexp(val, fmath::fr::reduced, fmath::fr::fpexp);
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
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<double, A> val) noexcept {
        auto const decomp =
            fmath::frexp(val, fmath::fr::reduced, fmath::fr::fpexp);
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

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              log2(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return log2(internal::abi<T>, val);
                } else {
                    return fallback(val);
                }
            } else {
                return log2(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::log2_t log2{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

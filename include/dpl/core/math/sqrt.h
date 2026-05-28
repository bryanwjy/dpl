// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/accuracy.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/rsqrt2.h"
#include "dpl/core/math/isfinite.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

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
    { sqrt(val) } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_sqrt = unqualified_extended_sqrt<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<sqrt_t, canonical_type_t<T>>);

struct sqrt_t {
private:
    template <simd_element E, simd_abi A>
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
    requires unqualified_sqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_sqrt<T>) {
            return sqrt(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sqrt_t sqrt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

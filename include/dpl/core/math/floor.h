// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/copysign.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

struct floor_t;

template <typename T>
concept unqualified_canonical_floor = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_neg_inf)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_floor = requires(T val) {
    {
        round(val, rounding::to_neg_inf)
    } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_floor = unqualified_extended_floor<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<floor_t, canonical_type_t<T>>);

template <typename T>
concept unqualified_canonical_floorne = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_neg_inf | rounding::no_exc)
    } -> equivalent_simd_as<T>;
};

template <typename T>
concept unqualified_extended_floorne = requires(T val) {
    {
        round(val, rounding::to_neg_inf | rounding::no_exc)
    } -> equivalent_simd_as<T>;
};

template <typename T>
concept unqualified_floorne = unqualified_extended_floor<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<floor_t, canonical_type_t<T>, rounding::no_exc_t>);

struct floor_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto fr = finite - dx::trunc(finite);
        fr += dx::select(fr > dx::zero, dx::broadcast<E, A>(dx::one), dx::zero);
        return dx::select(isfinite && dx::abs(val) < mx::maxint,
            dx::copysign(finite - fr, finite), val);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_floor<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val, rounding::to_neg_inf);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_floor<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return round(internal::abi<A>, val, rounding::to_neg_inf);
    }

    template <extended_vector T>
    requires unqualified_floor<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_floor<T>) {
            return round(val, rounding::to_neg_inf);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_canonical_floorne<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val,
                    rounding::to_neg_inf | rounding::no_exc);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_floorne<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return round(
            internal::abi<A>, val, rounding::to_neg_inf | rounding::no_exc);
    }

    template <extended_vector T>
    requires unqualified_floorne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_extended_floor<T>) {
            return round(val, rounding::to_neg_inf | rounding::no_exc);
        } else {
            return operator()(dx::to_canonical(val), rounding::no_exc);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::floor_t floor{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

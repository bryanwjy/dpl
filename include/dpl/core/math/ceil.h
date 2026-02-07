// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/copysign.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/abs.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

void ceil(...) noexcept = delete;

template <typename T>
concept unqualified_ceil =
    requires(T val) { round(internal::abi<T>, val, rounding::to_pos_inf); };

template <typename T>
concept unqualified_ceil_noexc = requires(T val) {
    round(internal::abi<T>, val, rounding::to_pos_inf | rounding::no_exc);
};

struct ceil_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::bit_keep(isfinite, val);
        auto fr = finite - dx::trunc(finite);
        fr -= dx::bit_keep(fr > dx::zero, dx::one_v<decltype(val)>);
        return dx::select(isfinite && dx::abs(val) < mx::maxint,
            dx::copysign(finite - fr, finite), val);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (unqualified_ceil<T>) {
            if not consteval {
                return round(internal::abi<T>, val, rounding::to_pos_inf);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T>) && unqualified_ceil<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept
        -> equivalent_simd_as<T> auto {
        return round(internal::abi<T>, val, rounding::to_pos_inf);
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T> && !unqualified_ceil<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        return operator()(dx::to_basic_type(val));
    }

    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_ceil_noexc<T>) {
            if not consteval {
                return round(internal::abi<T>, val,
                    rounding::to_pos_inf | rounding::no_exc);
            } else {
                return fallback(val);
            }
        } else {
            return fallback(val);
        }
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T>) && unqualified_ceil_noexc<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t) noexcept
        -> equivalent_simd_as<T> auto {
        return round(
            internal::abi<T>, val, rounding::to_pos_inf | rounding::no_exc);
    }

    template <floating_point_simd T>
    requires (!basic_simd_type<T> && !unqualified_ceil_noexc<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t tag) noexcept {
        return operator()(dx::to_basic_type(val), tag);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::ceil_t ceil{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

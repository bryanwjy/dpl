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

template <typename T>
concept unqualified_floor =
    requires(T val) { round(internal::abi<T>, val, rounding::to_neg_inf); };

template <typename T>
concept unqualified_floor_noexc = requires(T val) {
    round(internal::abi<T>, val, rounding::to_neg_inf | rounding::no_exc);
};

struct floor_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::bit_keep(isfinite, val);
        auto fr = finite - dx::trunc(finite);
        fr += dx::bit_keep(fr > dx::zero, dx::one_v<decltype(val)>);
        return dx::select(isfinite && dx::abs(val) < mx::maxint,
            dx::copysign(finite - fr, finite), val);
    }

public:
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (unqualified_floor<T>) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return round(internal::abi<T>, val, rounding::to_neg_inf);
                } else {
                    return fallback(val);
                }
            } else {
                return round(internal::abi<T>, val, rounding::to_neg_inf);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, rounding::no_exc_t tag [[maybe_unused]]) noexcept {
        if constexpr (unqualified_floor_noexc<T>) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return round(internal::abi<T>, val,
                        rounding::to_neg_inf | rounding::no_exc);
                } else {
                    return fallback(val);
                }
            } else {
                return round(internal::abi<T>, val,
                    rounding::to_neg_inf | rounding::no_exc);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val), tag);
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

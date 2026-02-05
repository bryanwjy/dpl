// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/operations/abs.h"
#  include "dpl/core/operations/bitwise.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void copysign(...) noexcept = delete;

template <typename L, typename R>
concept unqualified_copysign = requires(
    L lhs, R rhs) { copysign(internal::abi<common_abi_t<L, R>>, lhs, rhs); };

struct copysign_t {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> fallback(
        basic_simd<E, A> magnitude, basic_simd<E, A> sign) noexcept {
        return dx::bwor(dx::abs(magnitude), dx::bwand(dx::msb, sign));
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(T magnitude, T sign) noexcept {
        if constexpr (unqualified_copysign<T, T>) {
            if not consteval {
                return copysign(internal::abi<T>, magnitude, sign);
            } else {
                return fallback(magnitude, sign);
            }
        }

        return fallback(magnitude, sign);
    }

    template <floating_point_simd L, common_arithmetic_simd_with<L> R>
    requires only_unqualified<L, R> && unqualified_copysign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(L magnitude, R sign) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        return copysign(internal::abi<common_abi_t<L, R>>, magnitude, sign);
    }

    template <floating_point_simd L, common_arithmetic_simd_with<L> R>
    requires (!basic_simd_type<L> || !basic_simd_type<R> ||
                 !same_abi_simd_as<L, R>) &&
        (!unqualified_copysign<L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        L magnitude, R sign) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        return operator()(
            dx::to_basic_type(magnitude), dx::to_basic_type(sign));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::copysign_t copysign{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

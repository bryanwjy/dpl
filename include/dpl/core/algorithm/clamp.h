// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/math/fma.h"
#  include "dpl/core/operations/minmax.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void clamp(...) noexcept = delete;
template <typename A, typename L, typename M, typename R>
concept unqualified_clamp =
    requires(L val, M low, R high) { clamp(internal::abi<A>, val, low, high); };

struct clamp_t : fma_base<clamp_t> {
private:
    friend fma_base<clamp_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires unqualified_clamp<A, TA, TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA start, TB end, TC scale) noexcept {
        return clamp(abi, start, end, scale);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd<E, A> val,
        basic_simd<E, A> low, basic_simd<E, A> high) noexcept {
        return dx::min(high, dx::max(val, low));
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, T low, T high) noexcept {
        if constexpr (unqualified_clamp<T, T, T, T>) {
            if consteval {
                return fallback(val, low, high);
            } else {
                return clamp(internal::abi<T>, val, low, high);
            }
        } else {
            return fallback(val, low, high);
        }
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C> && only_unqualified_ternary<A, B, C> &&
        unqualified_clamp<common_abi_t<A, B, C>, A, B, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(A val, B low, C high) noexcept
        -> common_arithmetic_simd_with<common_arithmetic_simd_t<A, B, C>> auto {
        return clamp(internal::abi<common_abi_t<A, B, C>>, val, low, high);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C> &&
        (!basic_simd_type<A> || !basic_simd_type<B> || !basic_simd_type<C>) &&
        (!unqualified_clamp<common_abi_t<A, B, C>, A, B, C>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(A val, B low, C high) noexcept {
        return operator()(dx::to_basic_type(val), dx::to_basic_type(low),
            dx::to_basic_type(high));
    }

    using fma_base<clamp_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::clamp_t clamp{};
}

DPL_DEFAULT_NAMESPACE_END

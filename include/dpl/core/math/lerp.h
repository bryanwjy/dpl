// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/math/fma.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void lerp(...) noexcept = delete;
template <typename A, typename L, typename M, typename R>
concept unqualified_lerp =
    requires(L a, M b, R c) { lerp(internal::abi<A>, a, b, c); };

struct lerp_t : fma_base<lerp_t> {
private:
    friend fma_base<lerp_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    requires unqualified_lerp<A, TA, TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA start, TB end, TC scale) noexcept {
        return lerp(internal::abi<A>, start, end, scale);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd<E, A> start,
        basic_simd<E, A> end, basic_simd<E, A> scale) noexcept {
        return dx::fmadd(scale, dx::sub(end, start), start);
    }

public:
    template <basic_simd_type T>
    requires floating_point_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T a, T b, T c) noexcept {
        if constexpr (unqualified_lerp<T, T, T, T>) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return lerp(internal::abi<T>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C> && only_unqualified_ternary<A, B, C> &&
        unqualified_lerp<common_abi_t<A, B, C>, A, B, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept
        -> common_arithmetic_simd_with<common_arithmetic_simd_t<A, B, C>> auto {
        return lerp(internal::abi<common_abi_t<A, B, C>>, a, b, c);
    }

    template <simd_type A, common_arithmetic_simd_with<A> B,
        common_arithmetic_simd_with<common_arithmetic_simd_t<A, B>> C>
    requires floating_point_simd<A> && floating_point_simd<B> &&
        floating_point_simd<C> &&
        (!basic_simd_type<A> || !basic_simd_type<B> || !basic_simd_type<C>) &&
        (!unqualified_lerp<common_abi_t<A, B, C>, A, B, C>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(A a, B b, C c) noexcept {
        return operator()(
            dx::to_basic_type(a), dx::to_basic_type(b), dx::to_basic_type(c));
    }

    using fma_base<lerp_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::lerp_t lerp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

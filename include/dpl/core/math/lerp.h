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
template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_lerp = requires(L a, M b, R c) {
    { lerp(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};

struct lerp_t : fma_base<lerp_t> {
private:
    friend fma_base<lerp_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA start, TB end, TC scale) noexcept
    requires requires {
        {
            lerp(internal::abi<A>, start, end, scale)
        } -> floating_point_simd_with_abi<A>;
    }
    {
        return lerp(internal::abi<A>, start, end, scale);
    }

    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd<E, A> start,
        basic_simd<E, A> end, basic_simd<E, A> scale) noexcept {
        return dx::fmadd(scale, dx::subtract(end, start), start);
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_lerp<TA, TB, TC, A>) {
            if constexpr (basic_simd_type<TA> && basic_simd_type<TB> &&
                basic_simd_type<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return lerp(internal::abi<A>, a, b, c);
                }
            } else {
                return lerp(internal::abi<A>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TA> && basic_simd_type<TB> &&
            basic_simd_type<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(dx::to_basic_type(a), dx::to_basic_type(b),
                dx::to_basic_type(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_lerp<TA, TB, TC> ||
            unqualified_lerp<basic_type_t<TA>, basic_type_t<TB>,
                basic_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_lerp<TA, TB, TC>) {
            return lerp(internal::abi<A>, a, b, c);
        } else {
            return lerp(internal::abi<A>, dx::to_basic_type(a),
                dx::to_basic_type(b), dx::to_basic_type(c));
        }
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
